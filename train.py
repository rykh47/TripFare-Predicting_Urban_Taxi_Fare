import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from tqdm import tqdm  

from src.data import load_dataset
from src.features import engineer_features
from src.preprocess import remove_outliers_iqr, build_preprocessor
from src.models import get_models, get_param_grids, evaluate_regression


ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def main(args):
    df = load_dataset(args.data_path)

    # Basic filtering: drop obviously invalid rows
    if "total_amount" not in df.columns:
        raise ValueError("Dataset must contain 'total_amount' target column")

    # Engineer features
    df_fe = engineer_features(df)

    # Remove negative or zero target
    df_fe = df_fe[df_fe["total_amount"].astype(float) > 0]

    # Outlier handling on key numeric columns if present
    outlier_cols = [
        col
        for col in ["fare_amount", "trip_distance_km", "trip_duration_min", "total_amount"]
        if col in df_fe.columns
    ]
    df_fe = remove_outliers_iqr(df_fe, outlier_cols, iqr_multiplier=3.0)

    # Select features (drop target and obviously leakage columns if any)
    feature_df = df_fe.drop(columns=["total_amount"], errors="ignore")
    target = df_fe["total_amount"].astype(float)

    # Optionally sample for faster experimentation
    if args.sample and len(feature_df) > args.sample:
        feature_df = feature_df.sample(args.sample, random_state=42)
        target = target.loc[feature_df.index]

    X_train, X_test, y_train, y_test = train_test_split(
        feature_df, target, test_size=0.2, random_state=42
    )

    preprocessor, num_cols, cat_cols = build_preprocessor(X_train)
    models = get_models()

    results = {}
    best_name = None
    best_rmse = float("inf")
    best_pipeline = None

    # Add tqdm loading bar for model training
    for name, model in tqdm(models.items(), desc="Training models"):
        # Reduce sample size for SVR to avoid memory issues
        if name == "svr":
            X_train_svr = X_train.sample(min(5000, len(X_train)), random_state=42)
            y_train_svr = y_train.loc[X_train_svr.index]
            pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
            pipe.fit(X_train_svr, y_train_svr)
        else:
            pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
            pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        r2, mse, mae = evaluate_regression(y_test, preds)
        rmse = float(np.sqrt(mse))
        results[name] = {"r2": float(r2), "mse": float(mse), "rmse": rmse, "mae": float(mae)}
        if rmse < best_rmse:
            best_rmse = rmse
            best_name = name
            best_pipeline = pipe

    # Optional hyperparameter tuning on best candidate if enabled
    if args.tune and best_name in get_models().keys():
        param_grids = get_param_grids()
        if best_name in param_grids:
            search = RandomizedSearchCV(
                estimator=Pipeline(steps=[("preprocess", preprocessor), ("model", get_models()[best_name])]),
                param_distributions={f"model__{k}": v for k, v in param_grids[best_name].items()},
                n_iter=args.n_iter,
                scoring="neg_root_mean_squared_error",
                cv=3,
                n_jobs=-1,
                random_state=42,
                verbose=1,
            )
            search.fit(X_train, y_train)
            tuned_pipe = search.best_estimator_
            tuned_preds = tuned_pipe.predict(X_test)
            r2, mse, mae = evaluate_regression(y_test, tuned_preds)
            rmse = float(np.sqrt(mse))
            results[f"{best_name}_tuned"] = {
                "r2": float(r2),
                "mse": float(mse),
                "rmse": rmse,
                "mae": float(mae),
                "best_params": search.best_params_,
            }
            if rmse <= best_rmse:
                best_rmse = rmse
                best_name = f"{best_name}_tuned"
                best_pipeline = tuned_pipe

    # Save artifacts
    with open(ARTIFACTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    joblib.dump(best_pipeline, ARTIFACTS_DIR / "best_model.pkl")

    print("Best model:", best_name)
    print("Metrics:", json.dumps(results.get(best_name, {}), indent=2))
    print("All results saved to artifacts/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="taxi_fare.csv")
    parser.add_argument("--sample", type=int, default=150000)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--n-iter", type=int, default=20)
    args = parser.parse_args()
    main(args)