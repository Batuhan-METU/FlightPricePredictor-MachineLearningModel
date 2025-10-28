# src/model_training.py
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
import joblib

def train_decision_tree(X_train, y_train):
    model = DecisionTreeRegressor(random_state=15)
    model.fit(X_train, y_train)
    return model


def tune_hyperparameters(X_train, y_train):
    params = {
        "criterion": ["squared_error", "friedman_mse", "poisson", "absolute_error"],
        "max_depth": [10, 15, 20],
        "max_features": ["log2", "sqrt"],
    }

    search = RandomizedSearchCV(
        estimator=DecisionTreeRegressor(random_state=15),
        cv=5,
        param_distributions=params,
        n_jobs=-1,
    )

    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_


def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    return {"R2": r2, "MAE": mae, "MSE": mse}


def save_model(model, preprocessor, path):
    joblib.dump({"model": model, "preprocessor": preprocessor}, path)
