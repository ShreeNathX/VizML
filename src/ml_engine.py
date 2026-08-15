import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB


class MLEngine:
    @staticmethod
    def auto_detect_task(df: pd.DataFrame, target_col: str) -> Tuple[str, str]:
        if target_col not in df.columns:
            return "Classification", "Target column not found in dataset."

        col_data = df[target_col].dropna()
        if len(col_data) == 0:
            return "Classification", "Target column contains only null values."

        if not pd.api.types.is_numeric_dtype(col_data) or col_data.dtype == bool:
            return "Classification", f"Target '{target_col}' is non-numeric ({col_data.dtype})."

        is_float = pd.api.types.is_float_dtype(col_data)
        has_decimals = False
        if is_float:
            try:
                has_decimals = bool((col_data % 1 != 0).any())
            except Exception:
                has_decimals = True

        if has_decimals:
            return "Regression", f"Target '{target_col}' has continuous float values."

        n_unique = col_data.nunique()
        if n_unique <= 15:
            return "Classification", f"Target '{target_col}' has {n_unique} discrete unique values."

        return "Regression", f"Target '{target_col}' has continuous numeric values ({n_unique} unique)."

    @staticmethod
    def preprocess(
        df: pd.DataFrame,
        target_col: str,
        selected_features: List[str],
        task_type: str
    ) -> Dict[str, Any]:
        df_clean = df.copy()
        df_clean = df_clean.dropna(subset=[target_col])

        if len(df_clean) == 0:
            raise ValueError("No valid rows remaining after dropping missing target values.")

        y_raw = df_clean[target_col]
        le_target = None

        if task_type == "Classification":
            if not pd.api.types.is_numeric_dtype(y_raw) or y_raw.dtype == bool:
                le_target = LabelEncoder()
                y = le_target.fit_transform(y_raw.astype(str))
            else:
                le_target = LabelEncoder()
                y = le_target.fit_transform(y_raw.values)
        else:
            y_numeric = pd.to_numeric(y_raw, errors="coerce")
            valid_mask = y_numeric.notna()
            df_clean = df_clean[valid_mask]
            y = y_numeric[valid_mask].values.astype(float)
            if len(y) == 0:
                raise ValueError(f"Target column '{target_col}' could not be coerced to float for Regression.")

        feats = [f for f in selected_features if f in df_clean.columns and f != target_col]
        if not feats:
            raise ValueError("No valid feature columns selected.")

        X_df = df_clean[feats].copy()

        datetime_cols = []
        for col in X_df.columns:
            if pd.api.types.is_datetime64_any_dtype(X_df[col]):
                datetime_cols.append(col)
            elif X_df[col].dtype == object:
                sample = X_df[col].dropna().head(20)
                if len(sample) > 0:
                    try:
                        parsed = pd.to_datetime(sample, errors="coerce")
                        if parsed.notna().sum() / len(sample) > 0.8:
                            datetime_cols.append(col)
                    except Exception:
                        pass

        for col in datetime_cols:
            dt_series = pd.to_datetime(X_df[col], errors="coerce")
            X_df[f"{col}_year"] = dt_series.dt.year.fillna(dt_series.dt.year.median() if dt_series.dt.year.notna().any() else 2000)
            X_df[f"{col}_month"] = dt_series.dt.month.fillna(1)
            X_df[f"{col}_day"] = dt_series.dt.day.fillna(1)
            X_df[f"{col}_dayofweek"] = dt_series.dt.dayofweek.fillna(0)
            X_df.drop(columns=[col], inplace=True)

        cols_to_drop = []
        for col in X_df.columns:
            nun = X_df[col].nunique(dropna=True)
            if nun <= 1:
                cols_to_drop.append(col)
            elif X_df[col].dtype == object and len(X_df) > 20 and (nun / len(X_df)) > 0.95:
                cols_to_drop.append(col)

        if cols_to_drop:
            X_df.drop(columns=cols_to_drop, inplace=True)

        if X_df.shape[1] == 0:
            raise ValueError("All selected features were constant or high-cardinality ID columns.")

        numeric_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = [c for c in X_df.columns if c not in numeric_cols]

        numeric_imputers = {}
        for col in numeric_cols:
            med = X_df[col].median()
            val = med if pd.notna(med) else 0.0
            numeric_imputers[col] = val
            X_df[col] = X_df[col].fillna(val)

        categorical_imputers = {}
        onehot_encoders = {}
        label_encoders = {}
        encoded_feature_names = []

        processed_dfs = []
        if numeric_cols:
            processed_dfs.append(X_df[numeric_cols])
            encoded_feature_names.extend(numeric_cols)

        for col in categorical_cols:
            mode_val = X_df[col].mode(dropna=True)
            fill_val = mode_val.iloc[0] if len(mode_val) > 0 else "Missing"
            categorical_imputers[col] = fill_val
            series_filled = X_df[col].fillna(fill_val).astype(str)

            nunique = series_filled.nunique()
            if nunique <= 15:
                dummies = pd.get_dummies(series_filled, prefix=col, drop_first=False, dtype=float)
                onehot_encoders[col] = dummies.columns.tolist()
                processed_dfs.append(dummies)
                encoded_feature_names.extend(dummies.columns.tolist())
            else:
                le = LabelEncoder()
                encoded_vals = le.fit_transform(series_filled)
                label_encoders[col] = le
                df_encoded = pd.DataFrame({col: encoded_vals}, index=X_df.index)
                processed_dfs.append(df_encoded)
                encoded_feature_names.append(col)

        if not processed_dfs:
            raise ValueError("Failed to create processed feature matrices.")

        X_processed = pd.concat(processed_dfs, axis=1)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_processed.values)

        pipeline_meta = {
            "target_col": target_col,
            "task_type": task_type,
            "raw_features": feats,
            "encoded_feature_names": encoded_feature_names,
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            "datetime_cols": datetime_cols,
            "dropped_cols": cols_to_drop,
            "numeric_imputers": numeric_imputers,
            "categorical_imputers": categorical_imputers,
            "onehot_encoders": onehot_encoders,
            "label_encoders": label_encoders,
            "le_target": le_target,
            "scaler": scaler,
            "X_processed_df": X_processed,
            "X_scaled": X_scaled,
            "y": y,
        }

        return pipeline_meta

    @staticmethod
    def safe_split_and_cv(
        X_scaled: np.ndarray,
        y: np.ndarray,
        task_type: str,
        test_size_pct: float,
        cv_folds: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any, int, Optional[str]]:
        test_size = max(0.1, min(0.4, test_size_pct / 100.0))
        warning_msg = None
        stratify = None
        cv_strategy = cv_folds

        if task_type == "Classification":
            unique_classes, class_counts = np.unique(y, return_counts=True)
            min_class_cnt = int(class_counts.min()) if len(class_counts) > 0 else 0

            if len(unique_classes) < 2:
                raise ValueError("Target column has only 1 unique class — cannot train a classifier.")

            if min_class_cnt >= 2 and min_class_cnt >= int(1.0 / test_size):
                stratify = y
            else:
                warning_msg = (
                    f"Stratified split disabled: smallest class has only {min_class_cnt} sample(s), "
                    "which is too small for stratification."
                )

            if min_class_cnt < cv_folds:
                safe_cv = max(2, min_class_cnt)
                cv_strategy = safe_cv
                msg = f"Adjusted cross-validation folds from {cv_folds} to {safe_cv} due to class count."
                warning_msg = f"{warning_msg} {msg}" if warning_msg else msg

            if stratify is not None:
                cv_splitter = StratifiedKFold(n_splits=cv_strategy, shuffle=True, random_state=42)
            else:
                cv_splitter = KFold(n_splits=cv_strategy, shuffle=True, random_state=42)
        else:
            cv_strategy = max(2, min(cv_folds, len(y)))
            cv_splitter = KFold(n_splits=cv_strategy, shuffle=True, random_state=42)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42, stratify=stratify
        )

        return X_train, X_test, y_train, y_test, cv_splitter, cv_strategy, warning_msg

    @staticmethod
    def get_model_dictionary(task_type: str) -> Dict[str, Any]:
        if task_type == "Classification":
            return {
                "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
                "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
                "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
                "Decision Tree": DecisionTreeClassifier(random_state=42),
                "K-Nearest Neighbors": KNeighborsClassifier(),
                "SVM (RBF Kernel)": SVC(kernel="rbf", probability=True, random_state=42),
                "Naive Bayes": GaussianNB(),
            }
        else:
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
        progress_callback: Optional[Any] = None
    ) -> Dict[str, Any]:
        if progress_callback:
            progress_callback(0.1, "Preprocessing dataset & engineering features...")

        prep = cls.preprocess(df, target_col, selected_features, task_type)

        if progress_callback:
            progress_callback(0.25, "Setting up cross-validation splits...")

        X_train, X_test, y_train, y_test, cv_splitter, cv_effective, cv_warning = cls.safe_split_and_cv(
            prep["X_scaled"], prep["y"], task_type, test_size, cv_folds
        )

        all_models = cls.get_model_dictionary(task_type)
        results = []
        errors = []

        total_models = max(1, len(selected_model_names))

        for idx, model_name in enumerate(selected_model_names):
            if model_name not in all_models:
                continue

            pct = 0.3 + (0.65 * (idx / total_models))
            if progress_callback:
                progress_callback(pct, f"Training {model_name}...")

            model = all_models[model_name]

            try:
                scoring = "accuracy" if task_type == "Classification" else "r2"
                cv_scores = cross_val_score(
                    model, X_train, y_train, cv=cv_splitter, scoring=scoring
                )
                cv_mean = float(cv_scores.mean())
                cv_std = float(cv_scores.std())

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                if task_type == "Classification":
                    acc = float(accuracy_score(y_test, y_pred))
                    f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
                    results.append({
                        "Model": model_name,
                        "Test Accuracy": round(acc, 4),
                        "F1 Score (weighted)": round(f1, 4),
                        "CV Score (mean)": round(cv_mean, 4),
                        "CV Score (std)": round(cv_std, 4),
                        "_model": model,
                        "_y_pred": y_pred,
                        "_cv_mean": cv_mean,
                    })
                else:
                    mse = float(mean_squared_error(y_test, y_pred))
                    rmse = float(np.sqrt(mse))
                    r2 = float(r2_score(y_test, y_pred))
                    results.append({
                        "Model": model_name,
                        "Test R²": round(r2, 4),
                        "RMSE": round(rmse, 4),
                        "CV Score (mean)": round(cv_mean, 4),
                        "CV Score (std)": round(cv_std, 4),
                        "_model": model,
                        "_y_pred": y_pred,
                        "_y_test": y_test,
                        "_cv_mean": cv_mean,
                    })
            except Exception as e:
                errors.append(f"Model '{model_name}' failed to train: {str(e)}")

        if progress_callback:
            progress_callback(1.0, "Training complete!")

        if not results:
            error_str = " | ".join(errors) if errors else "No selected models completed successfully."
            raise RuntimeError(f"All model training attempts failed. Details: {error_str}")

        return {
            "results": results,
            "errors": errors,
            "pipeline": prep,
            "cv_effective_folds": cv_effective,
            "cv_warning": cv_warning,
            "X_test": X_test,
            "y_test": y_test,
            "task_type": task_type,
            "target_col": target_col,
        }

    @staticmethod
    def predict_single(
        pipeline_meta: Dict[str, Any],
        model: Any,
        input_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        df_row = pd.DataFrame([input_dict])

        for col in pipeline_meta["datetime_cols"]:
            if col in df_row.columns:
                dt_series = pd.to_datetime(df_row[col], errors="coerce")
                df_row[f"{col}_year"] = dt_series.dt.year.fillna(2000)
                df_row[f"{col}_month"] = dt_series.dt.month.fillna(1)
                df_row[f"{col}_day"] = dt_series.dt.day.fillna(1)
                df_row[f"{col}_dayofweek"] = dt_series.dt.dayofweek.fillna(0)
                df_row.drop(columns=[col], inplace=True)

        processed_parts = []

        numeric_cols = pipeline_meta["numeric_cols"]
        if numeric_cols:
            num_df = pd.DataFrame(index=df_row.index)
            for col in numeric_cols:
                fill_val = pipeline_meta["numeric_imputers"].get(col, 0.0)
                if col in df_row.columns:
                    val = pd.to_numeric(df_row[col].iloc[0], errors="coerce")
                    num_df[col] = [val if pd.notna(val) else fill_val]
                else:
                    num_df[col] = [fill_val]
            processed_parts.append(num_df)

        categorical_cols = pipeline_meta["categorical_cols"]
        for col in categorical_cols:
            fill_val = pipeline_meta["categorical_imputers"].get(col, "Missing")
            raw_val = str(df_row[col].iloc[0]) if col in df_row.columns and pd.notna(df_row[col].iloc[0]) else fill_val

            if col in pipeline_meta["onehot_encoders"]:
                cols_onehot = pipeline_meta["onehot_encoders"][col]
                oh_df = pd.DataFrame(0.0, index=df_row.index, columns=cols_onehot)
                matched_col = f"{col}_{raw_val}"
                if matched_col in oh_df.columns:
                    oh_df[matched_col] = 1.0
                processed_parts.append(oh_df)

            elif col in pipeline_meta["label_encoders"]:
                le = pipeline_meta["label_encoders"][col]
                try:
                    enc_val = le.transform([raw_val])[0]
                except Exception:
                    enc_val = 0
                lbl_df = pd.DataFrame({col: [enc_val]}, index=df_row.index)
                processed_parts.append(lbl_df)

        X_processed = pd.concat(processed_parts, axis=1)

        target_cols = pipeline_meta["encoded_feature_names"]
        X_aligned = X_processed.reindex(columns=target_cols, fill_value=0.0)

        scaler = pipeline_meta["scaler"]
        X_scaled = scaler.transform(X_aligned.values)

        raw_pred = model.predict(X_scaled)[0]

        proba_dict = None
        task_type = pipeline_meta["task_type"]
        le_target = pipeline_meta["le_target"]

        if task_type == "Classification":
            if hasattr(model, "predict_proba"):
                try:
                    probas = model.predict_proba(X_scaled)[0]
                    if le_target is not None:
                        classes = [str(c) for c in le_target.classes_]
                    else:
                        classes = [str(i) for i in range(len(probas))]
                    proba_dict = dict(zip(classes, [round(float(p), 4) for p in probas]))
                except Exception:
                    pass

            if le_target is not None:
                try:
                    display_pred = str(le_target.inverse_transform([int(raw_pred)])[0])
                except Exception:
                    display_pred = str(raw_pred)
            else:
                display_pred = str(raw_pred)
        else:
            display_pred = round(float(raw_pred), 4)

        return {
            "prediction": display_pred,
            "probabilities": proba_dict,
            "raw_prediction": raw_pred,
        }
