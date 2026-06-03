# ============================================================
# RIDESENSE AI - PIPELINE
# BEST MODEL: RandomForest + Winsorized Data
# ============================================================

import logging
import time
import os
import pickle

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
)


# ============================================================
# LOGGING SETUP
# ============================================================

os.makedirs("logs", exist_ok=True)
os.makedirs("artifacts", exist_ok=True)

logging.basicConfig(
    filename="logs/ridesense.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.Formatter.converter = time.localtime


# ============================================================
# SAVE PICKLE
# ============================================================

def save_pickle(obj, path):

    with open(path, "wb") as f:
        pickle.dump(obj, f)

    print(f"\nSaved: {path}")

    logging.info(f"Saved artifact: {path}")


# ============================================================
# 1. LOAD DATA
# ============================================================

def load_data(path):

    df = pd.read_csv(path)

    print("\n[DATA LOADED]", df.shape)

    logging.info(f"Data loaded — shape: {df.shape}")

    return df


# ============================================================
# 2. HANDLE MISSING VALUES
# ============================================================

def handle_missing(df):

    df = df.copy()

    for col in df.columns:

        if pd.api.types.is_numeric_dtype(df[col]):

            df[col] = df[col].fillna(
                df[col].median()
            )

        else:

            df[col] = df[col].fillna(
                df[col].mode()[0]
            )

    logging.info("Missing values handled.")

    print("\n[MISSING VALUES HANDLED]")

    return df


# ============================================================
# 3. WINSORIZATION
# ============================================================

def winsorize(df, num_cols):

    df = df.copy()

    for col in num_cols:

        q1 = df[col].quantile(0.25)

        q3 = df[col].quantile(0.75)

        iqr = q3 - q1

        lower = q1 - 1.5 * iqr

        upper = q3 + 1.5 * iqr

        df[col] = np.clip(
            df[col],
            lower,
            upper
        )

    logging.info("Winsorization completed.")

    print("\n[WINSORIZATION COMPLETED]")

    return df


# ============================================================
# 4. ENCODING
# ============================================================

def encode(df, target):

    X = df.drop(columns=[target])

    y = df[target]

    X = pd.get_dummies(
        X,
        drop_first=True
    )

    logging.info(
        f"Encoding done — Features: {X.shape[1]}"
    )

    print(
        f"\n[ENCODING DONE] Features: {X.shape[1]}"
    )

    return X, y


# ============================================================
# 5. TRAIN TEST SPLIT
# ============================================================

def split(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    logging.info(
        f"Train: {X_train.shape} | Test: {X_test.shape}"
    )

    print(
        f"\n[SPLIT DONE] Train: {X_train.shape} | Test: {X_test.shape}"
    )

    return X_train, X_test, y_train, y_test


# ============================================================
# 6. TRAIN BEST MODEL — RandomForest
# ============================================================

def train_model(X_train, y_train):

    model = RandomForestRegressor(

        n_estimators=200,

        n_jobs=-1,

        random_state=42
    )

    model.fit(X_train, y_train)

    logging.info(
        "RandomForest model trained."
    )

    print(
        "\n[MODEL TRAINED] RandomForestRegressor"
    )

    return model


# ============================================================
# 7. EVALUATE
# ============================================================

def evaluate(model, X_test, y_test):

    pred = model.predict(X_test)

    mse = mean_squared_error(y_test, pred)

    rmse = np.sqrt(mse)

    mae = mean_absolute_error(y_test, pred)

    mape = mean_absolute_percentage_error(
        y_test,
        pred
    )

    print("\n" + "="*60)

    print("   MODEL EVALUATION RESULTS")

    print("="*60)

    print(f"  MSE  : {mse:.2f}")

    print(f"  RMSE : {rmse:.2f}")

    print(f"  MAE  : {mae:.2f}")

    print(f"  MAPE : {mape*100:.2f}%")

    print("="*60)

    logging.info(
        f"Evaluation — RMSE: {rmse:.2f} | "
        f"MAE: {mae:.2f} | "
        f"MAPE: {mape*100:.2f}%"
    )

    return {

        "MSE": round(mse, 2),

        "RMSE": round(rmse, 2),

        "MAE": round(mae, 2),

        "MAPE": round(mape, 4),
    }


# ============================================================
# 8. FEATURE IMPORTANCE
# ============================================================

def feature_importance(model, columns):

    imp = pd.Series(

        model.feature_importances_,

        index=columns

    ).sort_values(ascending=False)

    print("\nTop 10 Features:")

    print(
        imp.head(10).to_string()
    )

    logging.info(
        "Feature importance computed."
    )

    return imp


# ============================================================
# 9. MAIN PIPELINE
# ============================================================

def run_pipeline():

    print("\n" + "="*60)

    print("   RIDESENSE AI — RIDE FARE PREDICTION PIPELINE")

    print("   Best Model : RandomForest")

    print("   Strategy   : Winsorized Data")

    print("="*60)

    logging.info("="*50)

    logging.info("RIDESENSE AI PIPELINE STARTED")

    logging.info("="*50)

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    df = load_data(
        "artifacts/ridesense_dataset.csv"
    )

    target = "ride_fare"

    # --------------------------------------------------------
    # HANDLE MISSING VALUES
    # --------------------------------------------------------

    df = handle_missing(df)

    # --------------------------------------------------------
    # NUMERICAL COLUMNS
    # --------------------------------------------------------

    num_cols = [

        c for c in df.select_dtypes(
            include=np.number
        ).columns

        if c != target
    ]

    # --------------------------------------------------------
    # WINSORIZATION
    # --------------------------------------------------------

    df = winsorize(df, num_cols)

    # --------------------------------------------------------
    # ENCODING
    # --------------------------------------------------------

    X, y = encode(df, target)

    # --------------------------------------------------------
    # SAVE TRAINING COLUMNS
    # --------------------------------------------------------

    save_pickle(

        X.columns.tolist(),

        "artifacts/model_columns.pkl"
    )

    # --------------------------------------------------------
    # SPLIT
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = split(
        X,
        y
    )

    # --------------------------------------------------------
    # TRAIN MODEL
    # --------------------------------------------------------

    model = train_model(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # EVALUATE
    # --------------------------------------------------------

    metrics = evaluate(
        model,
        X_test,
        y_test
    )

    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    feature_importance(
        model,
        X.columns
    )

    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------

    save_pickle(
        model,
        "artifacts/best_model.pkl"
    )

    print("\nPipeline completed successfully.")

    logging.info(
        "PIPELINE COMPLETED SUCCESSFULLY."
    )


# ============================================================
# RUN DIRECTLY
# ============================================================

if __name__ == "__main__":

    run_pipeline()