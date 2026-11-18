import unittest
import pickle
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.cleaner import calculate_purity_score
from src.ml_engine import MLEngine


class MLEngineTests(unittest.TestCase):
    def test_model_choices_match_task(self):
        classification = MLEngine.get_model_dictionary("Classification")
        regression = MLEngine.get_model_dictionary("Regression")
        self.assertIn("Logistic Regression", classification)
        self.assertNotIn("Logistic Regression", regression)
        self.assertIn("Linear Regression", regression)
        self.assertNotIn("Linear Regression", classification)

    def test_holdout_values_do_not_fit_preprocessor(self):
        df = pd.DataFrame({"feature": np.arange(50, dtype=float), "target": np.arange(50, dtype=float)})
        first_split = MLEngine.safe_split_and_cv(df[["feature"]], df["target"].to_numpy(), "Regression", 20, 5)
        test_indexes = first_split[1].index
        df.loc[test_indexes, "feature"] = 1_000_000

        metadata = MLEngine.preprocess(df, "target", ["feature"], "Regression")
        X_train, _, y_train, _, _, _, _ = MLEngine.safe_split_and_cv(
            metadata.pop("X"), metadata.pop("y"), "Regression", 20, 5
        )
        pipeline = MLEngine._build_pipeline(LinearRegression(), metadata).fit(X_train, y_train)
        imputer = pipeline.named_steps["preprocessor"].named_transformers_["numeric"].named_steps["imputer"]

        self.assertEqual(imputer.statistics_[0], X_train["feature"].median())
        self.assertNotEqual(imputer.statistics_[0], df["feature"].median())

    def test_string_dtype_dates_are_engineered(self):
        df = pd.DataFrame({
            "date": pd.Series(pd.date_range("2025-01-01", periods=40).strftime("%Y-%m-%d"), dtype="str"),
            "target": np.arange(40, dtype=float),
        })

        metadata = MLEngine.preprocess(df, "target", ["date"], "Regression")

        self.assertEqual(metadata["datetime_cols"], ["date"])
        self.assertIn("date_year", metadata["feature_cols"])

    def test_mixed_format_dates_are_engineered_without_missing_values(self):
        df = pd.DataFrame({
            "date": ["2025-01-01", "02/01/2025"] * 20,
            "target": np.arange(40, dtype=float),
        })

        metadata = MLEngine.preprocess(df, "target", ["date"], "Regression")

        self.assertEqual(metadata["datetime_cols"], ["date"])
        self.assertFalse(metadata["X"].isna().any().any())

    def test_all_null_string_feature_can_train_and_predict(self):
        df = pd.DataFrame({
            "empty": pd.Series([None] * 40, dtype="str"),
            "target": np.arange(40, dtype=float),
        })

        output = MLEngine.train_models(df, "target", ["empty"], "Regression", ["Linear Regression"])
        result = output["results"][0]
        prediction = MLEngine.predict_single(output["pipeline"], result["_model"], {"empty": None})

        self.assertIsInstance(prediction["prediction"], float)

    def test_knn_uses_a_valid_neighbor_count_for_small_cv_folds(self):
        df = pd.DataFrame({
            "feature": np.arange(8),
            "target": ["a"] * 4 + ["b"] * 4,
        })

        output = MLEngine.train_models(
            df, "target", ["feature"], "Classification", ["K-Nearest Neighbors"], cv_folds=5,
        )

        self.assertTrue(np.isfinite(output["results"][0]["_cv_mean"]))

        regression = pd.DataFrame({"feature": np.arange(8), "target": np.arange(8, dtype=float)})
        regression_output = MLEngine.train_models(
            regression, "target", ["feature"], "Regression", ["K-Nearest Neighbors"], cv_folds=5,
        )
        self.assertTrue(np.isfinite(regression_output["results"][0]["_cv_mean"]))

    def test_svm_trains_on_small_classification_dataset(self):
        for rows_per_class in (3, 4):
            with self.subTest(rows_per_class=rows_per_class):
                df = pd.DataFrame({
                    "feature": np.arange(rows_per_class * 2),
                    "target": ["a"] * rows_per_class + ["b"] * rows_per_class,
                })
                output = MLEngine.train_models(
                    df, "target", ["feature"], "Classification", ["SVM (RBF Kernel)"], cv_folds=5,
                )
                self.assertTrue(np.isfinite(output["results"][0]["_cv_mean"]))
                prediction = MLEngine.predict_single(
                    output["pipeline"], output["results"][0]["_model"], {"feature": 0},
                )
                self.assertEqual(prediction["probabilities"] is not None, rows_per_class == 4)

    def test_non_finite_cross_validation_score_is_rejected(self):
        df = pd.DataFrame({"feature": np.arange(40), "target": np.arange(40, dtype=float)})
        with patch("src.ml_engine.cross_val_score", return_value=np.array([np.nan])):
            with self.assertRaisesRegex(RuntimeError, "non-finite scores"):
                MLEngine.train_models(df, "target", ["feature"], "Regression", ["Linear Regression"])

    def test_all_models_train_and_unseen_category_predicts(self):
        rows = 80
        df = pd.DataFrame({
            "number": np.arange(rows, dtype=float),
            "category": [f"group_{i % 20}" for i in range(rows)],
            "class": ["yes" if i % 2 else "no" for i in range(rows)],
            "value": np.arange(rows, dtype=float) * 0.5 + np.sin(np.arange(rows)),
        })

        classification = MLEngine.train_models(
            df, "class", ["number", "category"], "Classification",
            list(MLEngine.get_model_dictionary("Classification")), cv_folds=3,
        )
        self.assertEqual(len(classification["results"]), 7)
        logistic = next(r for r in classification["results"] if r["Model"] == "Logistic Regression")
        features = logistic["_model"].named_steps["preprocessor"].get_feature_names_out()
        self.assertGreater(sum("category_" in name for name in features), 10)
        exported_model = pickle.loads(pickle.dumps(logistic["_model"]))
        prediction = MLEngine.predict_single(
            classification["pipeline"], exported_model, {"number": 1, "category": "unseen"}
        )
        self.assertIn(prediction["prediction"], {"yes", "no"})

        regression = MLEngine.train_models(
            df, "value", ["number", "category"], "Regression",
            list(MLEngine.get_model_dictionary("Regression")), cv_folds=3,
        )
        self.assertEqual(len(regression["results"]), 8)

    def test_tiny_classification_dataset_gets_clear_error(self):
        X = pd.DataFrame({"feature": [1, 2]})
        with self.assertRaisesRegex(ValueError, "Not enough examples per class"):
            MLEngine.safe_split_and_cv(X, np.array([0, 1]), "Classification", 20, 5)

    def test_purity_score_is_shared_and_bounded(self):
        df = pd.DataFrame({"a": [1, None], "b": [2, 3]})
        self.assertEqual(calculate_purity_score(1, df.size, 0, len(df)), 62.5)
        self.assertEqual(calculate_purity_score(0, 0, 0, 0), 100.0)


if __name__ == "__main__":
    unittest.main()
