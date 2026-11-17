import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import sklearn
from scipy import sparse
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def _to_dense(values):
    return values.toarray() if sparse.issparse(values) else values


class MLEngine:
    @staticmethod
    def auto_detect_task(df: pd.DataFrame, target_col: str) -> Tuple[str, str]:
        if target_col not in df.columns:
            return "Classification", "Target column not found in dataset."

        col_data = df[target_col].dropna()
        if len(col_data) == 0:
            return "Classification", "Target column contains only null values."

        if not pd.api.types.is_numeric_dtype(col_data) or pd.api.types.is_bool_dtype(col_data):
            return "Classification", f"Target '{target_col}' is non-numeric ({col_data.dtype})."

        if pd.api.types.is_float_dtype(col_data) and (col_data % 1 != 0).any():
            return "Regression", f"Target '{target_col}' has continuous float values."

        n_unique = col_data.nunique()
        if n_unique <= 15:
            return "Classification", f"Target '{target_col}' has {n_unique} discrete unique values."

        return "Regression", f"Target '{target_col}' has continuous numeric values ({n_unique} unique)."

    @staticmethod
    def _expand_datetime_features(X: pd.DataFrame, datetime_cols: List[str]) -> pd.DataFrame:
        X = X.copy()
        for col in datetime_cols:
            if col not in X:
                continue
            dates = pd.to_datetime(X[col], errors="coerce")
            for part in ("year", "month", "day", "dayofweek"):
                X[f"{col}_{part}"] = getattr(dates.dt, part)
            X.drop(columns=col, inplace=True)
        return X

    @classmethod
    def preprocess(
        cls,
        df: pd.DataFrame,
        target_col: str,
        selected_features: List[str],
        task_type: str,
    ) -> Dict[str, Any]:
        if task_type not in ("Classification", "Regression"):
            raise ValueError(f"Unsupported task type: {task_type}")
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset.")

        df_clean = df.dropna(subset=[target_col]).copy()
        if df_clean.empty:
            raise ValueError("No valid rows remaining after dropping missing target values.")

        y_raw = df_clean[target_col]
        le_target = None
        if task_type == "Classification":
            le_target = LabelEncoder()
            y = le_target.fit_transform(y_raw.astype(str) if not pd.api.types.is_numeric_dtype(y_raw) else y_raw)
        else:
            y_numeric = pd.to_numeric(y_raw, errors="coerce")
            valid = y_numeric.notna()
            df_clean = df_clean.loc[valid]
            y = y_numeric.loc[valid].to_numpy(dtype=float)
            if len(y) == 0:
                raise ValueError(f"Target column '{target_col}' could not be coerced to float for Regression.")

        raw_features = [c for c in selected_features if c in df_clean.columns and c != target_col]
        if not raw_features:
            raise ValueError("No valid feature columns selected.")

        X = df_clean[raw_features].copy()
        datetime_cols = []
        for col in X.columns:
            if pd.api.types.is_datetime64_any_dtype(X[col]):
                datetime_cols.append(col)
            elif X[col].dtype == object:
                sample = X[col].dropna().head(20)
                if not sample.empty:
                    parsed = pd.to_datetime(sample, errors="coerce")
                    if parsed.notna().mean() > 0.8:
                        datetime_cols.append(col)

        X = cls._expand_datetime_features(X, datetime_cols)
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = [c for c in X.columns if c not in numeric_cols]
        for col in categorical_cols:
            X[col] = X[col].map(lambda value: str(value) if pd.notna(value) else np.nan)

        return {
            "target_col": target_col,
            "task_type": task_type,
            "raw_features": raw_features,
            "feature_cols": X.columns.tolist(),
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            "datetime_cols": datetime_cols,
            "le_target": le_target,
            "sklearn_version": sklearn.__version__,
            "X": X,
            "y": y,
        }

    @staticmethod
    def safe_split_and_cv(
        X: pd.DataFrame,
        y: np.ndarray,
        task_type: str,
        test_size_pct: int,
        cv_folds: int,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, Any, int, Optional[str]]:
        test_size = max(0.1, min(0.4, test_size_pct / 100.0))
        test_count = math.ceil(len(y) * test_size)
        train_count = len(y) - test_count
        if task_type == "Classification":
            classes, counts = np.unique(y, return_counts=True)
            if len(classes) < 2:
                raise ValueError("Target column has only 1 unique class - cannot train a classifier.")
            if counts.min() < 2 or test_count < len(classes) or train_count < len(classes):
                raise ValueError(
                    "Not enough examples per class for a stratified holdout and cross-validation. "
                    "Add rows or reduce the number of classes."
                )
            stratify = y
        else:
            if test_count < 2:
                raise ValueError(
                    "Regression needs at least 2 holdout rows to calculate R². "
                    "Add more rows or increase the holdout split."
                )
            if train_count < 4:
                raise ValueError("Regression needs at least 4 training rows for 2-fold cross-validation.")
            stratify = None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=stratify
        )

        if task_type == "Classification":
            _, train_counts = np.unique(y_train, return_counts=True)
            folds = min(cv_folds, int(train_counts.min()))
            if folds < 2:
                raise ValueError("Each class needs at least 2 training examples for cross-validation.")
            splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        else:
            folds = min(cv_folds, len(y_train) // 2)
            splitter = KFold(n_splits=folds, shuffle=True, random_state=42)

        warning = (
            f"Adjusted cross-validation folds from {cv_folds} to {folds} to fit the training data."
            if folds != cv_folds
            else None
        )
        return X_train, X_test, y_train, y_test, splitter, folds, warning

    @staticmethod
    def get_model_dictionary(task_type: str) -> Dict[str, Any]:
        if task_type == "Classification":
            return {
                "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
                "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
                "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
                "Decision Tree": DecisionTreeClassifier(random_state=42),
                "K-Nearest Neighbors": KNeighborsClassifier(),
                "SVM (RBF Kernel)": CalibratedClassifierCV(SVC(kernel="rbf", random_state=42), cv=3),
                "Naive Bayes": GaussianNB(),
            }
        if task_type == "Regression":
            return {
                "Linear Regression": LinearRegression(),
                "Ridge Regression": Ridge(alpha=1.0, random_state=42),
                "Lasso Regression": Lasso(alpha=0.1, max_iter=2000, random_state=42),
                "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
                "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
                "Decision Tree": DecisionTreeRegressor(random_state=42),
                "K-Nearest Neighbors": KNeighborsRegressor(),
                "SVR (RBF Kernel)": SVR(kernel="rbf"),
            }
        raise ValueError(f"Unsupported task type: {task_type}")

    @staticmethod
    def _build_pipeline(model: Any, metadata: Dict[str, Any]) -> Pipeline:
        transformers = []
        if metadata["numeric_cols"]:
            transformers.append((
                "numeric",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                    ("scaler", StandardScaler()),
                ]),
                metadata["numeric_cols"],
            ))
        if metadata["categorical_cols"]:
            transformers.append((
                "categorical",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="constant", fill_value="Missing", keep_empty_features=True)),
                    ("onehot", OneHotEncoder(
                        handle_unknown="ignore", min_frequency=2, max_categories=100,
                        sparse_output=True,
                    )),
                ]),
                metadata["categorical_cols"],
            ))

        steps = [("preprocessor", ColumnTransformer(transformers))]
        if isinstance(model, (GradientBoostingClassifier, GradientBoostingRegressor, GaussianNB)):
            steps.append(("dense", FunctionTransformer(_to_dense, accept_sparse=True)))
        steps.append(("model", model))
        return Pipeline(steps)

    @classmethod
    def train_models(
        cls,
        df: pd.DataFrame,
        target_col: str,
        selected_features: List[str],
        task_type: str,
        selected_model_names: List[str],
        test_size: int = 20,
        cv_folds: int = 5,
        progress_callback: Optional[Any] = None,
    ) -> Dict[str, Any]:
        if not selected_model_names:
            raise ValueError("Select at least one model to train.")
        if progress_callback:
            progress_callback(0.1, "Preprocessing dataset & engineering features...")

        metadata = cls.preprocess(df, target_col, selected_features, task_type)
        X_train, X_test, y_train, y_test, cv_splitter, cv_effective, cv_warning = cls.safe_split_and_cv(
            metadata.pop("X"), metadata.pop("y"), task_type, test_size, cv_folds
        )
        all_models = cls.get_model_dictionary(task_type)
        unknown_models = set(selected_model_names) - set(all_models)
        if unknown_models:
            raise ValueError(f"Models do not match {task_type}: {', '.join(sorted(unknown_models))}.")

        results = []
        errors = []
        total_models = len(selected_model_names)
        for idx, model_name in enumerate(selected_model_names):
            if progress_callback:
                progress_callback(0.3 + 0.65 * idx / total_models, f"Training {model_name}...")
            model = cls._build_pipeline(all_models[model_name], metadata)
            try:
                scoring = "accuracy" if task_type == "Classification" else "r2"
                cv_scores = cross_val_score(model, X_train, y_train, cv=cv_splitter, scoring=scoring)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                result = {
                    "Model": model_name,
                    "CV Score (mean)": round(float(cv_scores.mean()), 4),
                    "CV Score (std)": round(float(cv_scores.std()), 4),
                    "_model": model,
                    "_y_pred": y_pred,
                    "_cv_mean": float(cv_scores.mean()),
                }
                if task_type == "Classification":
                    result["Test Accuracy"] = round(float(accuracy_score(y_test, y_pred)), 4)
                    result["F1 Score (weighted)"] = round(
                        float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4
                    )
                else:
                    result["Test R²"] = round(float(r2_score(y_test, y_pred)), 4)
                    result["RMSE"] = round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4)
                    result["_y_test"] = y_test
                results.append(result)
            except Exception as err:
                errors.append(f"Model '{model_name}' failed to train: {err}")

        if progress_callback:
            progress_callback(1.0, "Training complete!")
        if not results:
            detail = " | ".join(errors) if errors else "No selected models completed successfully."
            raise RuntimeError(f"All model training attempts failed. Details: {detail}")

        return {
            "results": results,
            "errors": errors,
            "pipeline": metadata,
            "cv_effective_folds": cv_effective,
            "cv_warning": cv_warning,
            "y_test": y_test,
            "task_type": task_type,
            "target_col": target_col,
        }

    @staticmethod
    def predict_single(
        pipeline_meta: Dict[str, Any],
        model: Pipeline,
        input_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        row = pd.DataFrame([input_dict]).reindex(columns=pipeline_meta["raw_features"])
        row = MLEngine._expand_datetime_features(row, pipeline_meta["datetime_cols"])
        row = row.reindex(columns=pipeline_meta["feature_cols"])
        for col in pipeline_meta["numeric_cols"]:
            row[col] = pd.to_numeric(row[col], errors="coerce")
        for col in pipeline_meta["categorical_cols"]:
            row[col] = row[col].map(lambda value: str(value) if pd.notna(value) else np.nan)

        raw_pred = model.predict(row)[0]
        task_type = pipeline_meta["task_type"]
        le_target = pipeline_meta["le_target"]
        proba_dict = None
        if task_type == "Classification":
            estimator = model.named_steps["model"]
            if hasattr(estimator, "predict_proba"):
                probabilities = model.predict_proba(row)[0]
                classes = le_target.inverse_transform(estimator.classes_.astype(int))
                proba_dict = {
                    str(label): round(float(probability), 4)
                    for label, probability in zip(classes, probabilities)
                }
            display_pred = str(le_target.inverse_transform([int(raw_pred)])[0])
        else:
            display_pred = round(float(raw_pred), 4)

        return {
            "prediction": display_pred,
            "probabilities": proba_dict,
            "raw_prediction": raw_pred,
        }
