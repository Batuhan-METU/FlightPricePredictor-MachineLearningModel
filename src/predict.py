# src/predict.py
import joblib
import pandas as pd

def load_model(path):
    data = joblib.load(path)
    model = data["model"]
    preprocessor = data["preprocessor"]
    return model, preprocessor


def predict_price(model, preprocessor, input_dict):
    df = pd.DataFrame([input_dict])
    X = preprocessor.transform(df)
    pred = model.predict(X)[0]
    return float(pred)
