import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
from src.ml_engine import MLEngine


def test_auto_detect_task():
    df = pd.DataFrame({
        "str_cat": ["A", "B", "A", "C"],
        "num_cat": [1, 2, 1, 2],
        "cont_num": [10.5, 20.3, 15.6, 99.1],
    })
    t1, _ = MLEngine.auto_detect_task(df, "str_cat")
    assert t1 == "Classification"

    t2, _ = MLEngine.auto_detect_task(df, "num_cat")
    assert t2 == "Classification"

    t3, _ = MLEngine.auto_detect_task(df, "cont_num")
    assert t3 == "Regression"


def test_preprocessing_and_train():
    df = pd.DataFrame({
        "target": ["Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No"],
        "age": [25, 30, np.nan, 45, 50, 22, 35, 40, 29, 31],
        "city": ["NYC", "LA", "NYC", "Chicago", "LA", "NYC", "Chicago", "LA", "NYC", "Chicago"],
        "date_str": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05",
                    "2023-01-06", "2023-01-07", "2023-01-08", "2023-01-09", "2023-01-10"],
        "id_col": [f"ID_{i}" for i in range(10)]
    })

    res = MLEngine.train_models(
        df=df,
        target_col="target",
        selected_features=["age", "city", "date_str", "id_col"],
        task_type="Classification",
        selected_model_names=["Logistic Regression", "Random Forest"],
        test_size=20,
        cv_folds=2,
    )

    assert "results" in res
    assert len(res["results"]) == 2
    assert "pipeline" in res

    # Test single prediction
    best_model = res["results"][0]["_model"]
    pipeline = res["pipeline"]
    sample_input = {"age": 28, "city": "NYC", "date_str": "2023-01-15"}
    pred_res = MLEngine.predict_single(pipeline, best_model, sample_input)

    assert "prediction" in pred_res
    assert pred_res["prediction"] in ["Yes", "No"]


def test_regression_pipeline():
    np.random.seed(42)
    df = pd.DataFrame({
        "price": np.random.uniform(100, 500, 30),
        "sqft": np.random.uniform(500, 2500, 30),
        "bedrooms": np.random.choice([1, 2, 3, 4], 30),
        "neighborhood": np.random.choice(["Downtown", "Suburbs", "Uptown"], 30)
    })

    res = MLEngine.train_models(
        df=df,
        target_col="price",
        selected_features=["sqft", "bedrooms", "neighborhood"],
        task_type="Regression",
        selected_model_names=["Linear Regression", "Random Forest"],
        test_size=20,
        cv_folds=3,
    )

    assert len(res["results"]) == 2
    best_model = res["results"][0]["_model"]
    pipeline = res["pipeline"]
    sample_input = {"sqft": 1200, "bedrooms": 2, "neighborhood": "Downtown"}
    pred_res = MLEngine.predict_single(pipeline, best_model, sample_input)

    assert "prediction" in pred_res
    assert isinstance(pred_res["prediction"], (int, float))


if __name__ == "__main__":
    print("Running MLEngine tests...")
    test_auto_detect_task()
    print("- test_auto_detect_task passed.")
    test_preprocessing_and_train()
    print("- test_preprocessing_and_train passed.")
    test_regression_pipeline()
    print("- test_regression_pipeline passed.")
    print("All MLEngine tests passed successfully!")
