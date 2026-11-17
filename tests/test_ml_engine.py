import unittest
import pickle

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
