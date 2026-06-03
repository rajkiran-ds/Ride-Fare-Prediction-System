# ============================================================
# RIDESENSE AI - EXPERIMENTATION PIPELINE
# REGRESSION PROJECT
# ============================================================

import logging
import time
import os
import pickle

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor
)

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error
)


# ============================================================
# LOGGING SETUP
# ============================================================

os.makedirs("logs", exist_ok=True)
os.makedirs("artifacts", exist_ok=True)

logging.basicConfig(
    filename="logs/ridesense_experimentation.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.Formatter.converter = time.localtime


# ============================================================
# 1. LOAD DATA
# ============================================================

def load_data(path):

    logging.info("Loading dataset")

    df = pd.read_csv(path)

    print("\n[DATA LOADED]")
    print("Shape:", df.shape)

    return df


# ============================================================
# 2. HANDLE MISSING VALUES
# ============================================================

def handle_missing(df):

    logging.info("Handling missing values")

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

    print("\n[MISSING VALUES HANDLED]")

    return df


# ============================================================
# 3. WINSORIZATION
# ============================================================

def winsorize(df, num_cols):

    logging.info("Applying winsorization")

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

    print("\n[WINSORIZATION COMPLETED]")

    return df


# ============================================================
# 4. LOG TRANSFORMATION
# ============================================================

def log_transform(df, num_cols):

    logging.info("Applying log transformation")

    df = df.copy()

    for col in num_cols:

        if (df[col] >= 0).all():

            df[col] = np.log1p(df[col])

    print("\n[LOG TRANSFORMATION COMPLETED]")

    return df


# ============================================================
# 5. FEATURE SELECTION
# ============================================================

def feature_selection(X, y):

    logging.info("Performing feature selection")

    selector_model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    selector_model.fit(X, y)

    importance = selector_model.feature_importances_

    threshold = np.mean(importance)

    selected_cols = X.columns[
        importance >= threshold
    ]

    print("\n[FEATURE SELECTION COMPLETED]")
    print("Selected Features:", len(selected_cols))

    return X[selected_cols]


# ============================================================
# 6. ENCODING
# ============================================================

def encode(df, target):

    logging.info("Encoding categorical variables")

    X = df.drop(columns=[target])

    y = df[target]

    X = pd.get_dummies(
        X,
        drop_first=True
    )

    return X, y


# ============================================================
# 7. TRAIN TEST SPLIT
# ============================================================

def split(X, y):

    logging.info("Performing train-test split")

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )


# ============================================================
# 8. SCALING
# ============================================================

def scale(X_train, X_test):

    logging.info("Scaling features")

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    print("\n[SCALING COMPLETED]")

    return X_train_scaled, X_test_scaled


# ============================================================
# 9. MODELS
# ============================================================

def get_models():

    return {

        "LinearRegression": LinearRegression(),

        "KNNRegressor": KNeighborsRegressor(),

        "DecisionTree": DecisionTreeRegressor(
            random_state=42
        ),

        "RandomForest": RandomForestRegressor(
            random_state=42
        ),

        "AdaBoost": AdaBoostRegressor(
            random_state=42
        )
    }


# ============================================================
# 10. EVALUATION
# ============================================================

def evaluate(
        model,
        X_train,
        X_test,
        y_train,
        y_test
):

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    mse = mean_squared_error(
        y_test,
        pred
    )

    rmse = np.sqrt(mse)

    mae = mean_absolute_error(
        y_test,
        pred
    )

    mape = mean_absolute_percentage_error(
        y_test,
        pred
    )

    result = {

        "MSE": round(mse, 2),

        "RMSE": round(rmse, 2),

        "MAE": round(mae, 2),

        "MAPE": round(mape, 4)
    }

    return result


# ============================================================
# 11. SAVE MODEL
# ============================================================

def save_model(model, path):

    logging.info("Saving model")

    with open(path, "wb") as f:

        pickle.dump(model, f)

    print(f"\n[MODEL SAVED] {path}")


# ============================================================
# 12. RUN EXPERIMENTATION PIPELINE
# ============================================================

def run_experimentation(path):

    logging.info("Experimentation pipeline started")

    # ========================================================
    # LOAD DATA
    # ========================================================

    df = load_data(path)

    target = "ride_fare"

    # ========================================================
    # HANDLE MISSING VALUES
    # ========================================================

    df = handle_missing(df)

    # ========================================================
    # NUMERICAL COLUMNS
    # ========================================================

    num_cols = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    num_cols = [
        col for col in num_cols
        if col != target
    ]

    # ========================================================
    # WINSORIZATION
    # ========================================================

    df_w = winsorize(
        df,
        num_cols
    )

    # ========================================================
    # LOG TRANSFORMATION
    # ========================================================

    df_l = log_transform(
        df_w,
        num_cols
    )

    # ========================================================
    # MODELS
    # ========================================================

    models = get_models()

    # ========================================================
    # BASE DATA EXPERIMENT
    # ========================================================

    X, y = encode(df, target)

    X_train, X_test, y_train, y_test = split(
        X,
        y
    )

    print("\n===== BASE DATA RESULTS =====")

    base_results = []

    for name, model in models.items():

        res = evaluate(
            model,
            X_train,
            X_test,
            y_train,
            y_test
        )

        res["Model"] = name

        base_results.append(res)

    base_df = pd.DataFrame(base_results)

    # ========================================================
    # SCALED DATA EXPERIMENT
    # ========================================================

    X, y = encode(df, target)

    X_train, X_test, y_train, y_test = split(
        X,
        y
    )

    X_train_scaled, X_test_scaled = scale(
        X_train,
        X_test
    )

    print("\n===== SCALED DATA RESULTS =====")

    scaled_results = []

    for name, model in models.items():

        res = evaluate(
            model,
            X_train_scaled,
            X_test_scaled,
            y_train,
            y_test
        )

        res["Model"] = name

        scaled_results.append(res)

    scaled_df = pd.DataFrame(scaled_results)

    # ========================================================
    # WINSORIZED DATA EXPERIMENT
    # ========================================================

    X, y = encode(df_w, target)

    X_train, X_test, y_train, y_test = split(
        X,
        y
    )

    X_train_scaled, X_test_scaled = scale(
        X_train,
        X_test
    )

    print("\n===== WINSORIZED DATA RESULTS =====")

    win_results = []

    for name, model in models.items():

        res = evaluate(
            model,
            X_train_scaled,
            X_test_scaled,
            y_train,
            y_test
        )

        res["Model"] = name

        win_results.append(res)

    win_df = pd.DataFrame(win_results)

    # ========================================================
    # LOG TRANSFORM DATA EXPERIMENT
    # ========================================================

    X, y = encode(df_l, target)

    X_train, X_test, y_train, y_test = split(
        X,
        y
    )

    X_train_scaled, X_test_scaled = scale(
        X_train,
        X_test
    )

    print("\n===== LOG TRANSFORM RESULTS =====")

    log_results = []

    for name, model in models.items():

        res = evaluate(
            model,
            X_train_scaled,
            X_test_scaled,
            y_train,
            y_test
        )

        res["Model"] = name

        log_results.append(res)

    log_df = pd.DataFrame(log_results)

    # ========================================================
    # FEATURE SELECTION EXPERIMENT
    # ========================================================

    X, y = encode(df, target)

    X = feature_selection(X, y)

    X_train, X_test, y_train, y_test = split(
        X,
        y
    )

    X_train_scaled, X_test_scaled = scale(
        X_train,
        X_test
    )

    print("\n===== FEATURE SELECTION RESULTS =====")

    fs_results = []

    best_model = None
    best_rmse = float("inf")

    for name, model in models.items():

        res = evaluate(
            model,
            X_train_scaled,
            X_test_scaled,
            y_train,
            y_test
        )

        res["Model"] = name

        fs_results.append(res)

        if res["RMSE"] < best_rmse:

            best_rmse = res["RMSE"]

            best_model = model

    fs_df = pd.DataFrame(fs_results)

    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    best_model.fit(
        X_train_scaled,
        y_train
    )

    save_model(
        best_model,
        "artifacts/ridesense_best_model.pkl"
    )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print("\n========== FINAL COMPARISON ==========\n")

    print(
        "\nBASE DATA\n",
        base_df.sort_values(
            "RMSE"
        )
    )

    print(
        "\nSCALED DATA\n",
        scaled_df.sort_values(
            "RMSE"
        )
    )

    print(
        "\nWINSORIZED DATA\n",
        win_df.sort_values(
            "RMSE"
        )
    )

    print(
        "\nLOG TRANSFORM DATA\n",
        log_df.sort_values(
            "RMSE"
        )
    )

    print(
        "\nFEATURE SELECTION DATA\n",
        fs_df.sort_values(
            "RMSE"
        )
    )

    logging.info(
        "Experimentation pipeline completed"
    )

    return (
        base_df,
        scaled_df,
        win_df,
        log_df,
        fs_df
    )


# ============================================================
# EXECUTION
# ============================================================

if __name__ == "__main__":

    run_experimentation(
        "artifacts/ridesense_dataset.csv"
    )