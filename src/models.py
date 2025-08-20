from typing import Dict, Tuple
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def get_models() -> Dict[str, object]:
    return {
        "linear": LinearRegression(),
        "ridge": Ridge(),
        "lasso": Lasso(max_iter=10000),
        "random_forest": RandomForestRegressor(n_jobs=-1, random_state=42),
        "gbm": GradientBoostingRegressor(random_state=42),
        "svr": SVR(kernel="linear"),  # Use linear kernel for stability
        "decision_tree": DecisionTreeRegressor(random_state=42),
    }


def get_param_grids() -> Dict[str, dict]:
    return {
        "Ridge": {"alpha": np.logspace(-3, 3, 13)},
        "Lasso": {"alpha": np.logspace(-4, 1, 10)},
        "RandomForest": {
            "n_estimators": [200, 300, 500],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        },
        "GradientBoosting": {
            "n_estimators": [200, 300],
            "learning_rate": [0.03, 0.05, 0.1],
            "max_depth": [2, 3, 4],
            "subsample": [0.7, 0.85, 1.0],
        },
    }


def evaluate_regression(y_true, y_pred) -> Tuple[float, float, float]:
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    return r2, mse, mae