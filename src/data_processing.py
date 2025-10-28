# src/data_processing.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

def load_data(path: str):

    df = pd.read_csv(path)

    if "Unnamed: 0" in df.columns:
        df.drop("Unnamed: 0", axis=1, inplace=True)

    if "flight" in df.columns:
        df.drop("flight", axis=1, inplace=True)

    df["stops"] = (
        df["stops"]
        .str.replace("zero", "0")
        .str.replace("one", "1")
        .str.replace("two_or_more", "2")
        .astype(int)
    )

    return df


def preprocess_data(df: pd.DataFrame):
    X = df.drop("price", axis=1)
    y = df["price"]

    categorical_cols = [
        "airline",
        "source_city",
        "departure_time",
        "arrival_time",
        "destination_city",
        "class",
    ]
    numeric_cols = ["stops", "duration", "days_left"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols)
        ],
        remainder="passthrough"
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=15
    )

    X_train_enc = preprocessor.fit_transform(X_train)
    X_test_enc = preprocessor.transform(X_test)

    encoded_cols = preprocessor.get_feature_names_out()

    return X_train_enc, X_test_enc, y_train, y_test, preprocessor, encoded_cols
