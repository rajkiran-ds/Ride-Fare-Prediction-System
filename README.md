# RideSense AI – Ride Fare Prediction System

## Problem Statement

RideSense AI is a ride-hailing fare prediction system designed to address inconsistencies in ride fare estimation caused by:

* Traffic conditions
* Weather changes
* Peak-hour demand
* Dynamic surge pricing
* Driver availability

Inaccurate fare estimation can lead to:

* Customer dissatisfaction
* Driver complaints
* Revenue inefficiencies

This project builds a Machine Learning regression system capable of predicting ride fares using historical ride data and operational parameters.


# Business Objective

* Improve fare estimation accuracy
* Reduce pricing inconsistencies
* Enhance customer experience
* Improve operational efficiency
* Build a scalable fare prediction pipeline
* Minimize prediction error using regression metrics

# Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Streamlit
* Docker
* Pickle
* Logging
* Git & GitHub


# Machine Learning Workflow

## Data Loading

Historical ride data loaded using Pandas.

## Missing Value Handling

* Median imputation for numerical columns
* Mode imputation for categorical columns

## Outlier Treatment

Winsorization applied using the IQR method to reduce the impact of extreme values.

## Log Transformation

Log transformation applied to numerical features for experimentation.

## Feature Encoding

Categorical variables encoded using one-hot encoding.

## Train-Test Split

Dataset split into training and testing sets.

## Model Experimentation

Multiple regression algorithms evaluated:

* Linear Regression
* KNeighbors Regressor
* Decision Tree Regressor
* Random Forest Regressor
* AdaBoost Regressor

## Feature Selection

Feature importance based selection performed using Random Forest.

## Model Evaluation

Models evaluated using:

* RMSE
* MAE
* MAPE

## Final Model

Selected Model:

* RandomForestRegressor

Final Strategy:

* Winsorized Dataset

# Features

## Numerical Features

* Trip Distance
* Trip Duration
* Surge Multiplier
* Fuel Price
* Demand Index
* Driver Rating
* Customer Rating
* Customer Loyalty Score
* Phone Battery Level

## Categorical Features

* Traffic Level
* Vehicle Type
* Weather Condition
* Peak Hour
* Pickup Zone
* Music Preference


# Streamlit Application

The project includes an interactive Streamlit web application that allows users to:

* Enter ride details
* Select ride conditions
* Predict ride fare in real time

# Docker Support

## Build Docker Image

docker build -t ridesense-app .

## Run Docker Container

docker run  ridesense-ai-app


# Project Structure

RideSense_AI/
│
├── artifacts/
│   ├── best_model.pkl
│   ├── model_columns.pkl
│   └── ridesense_dataset.csv
│
├── images/
│
├── logs/
│   ├── ridesense.log
│   └── ridesense_experimentation.log
│
├── app.py
├── pipeline.py
├── experimentation.py
├── Dockerfile
├── requirements.txt
├── README.md
└── .gitignore


# Run Project

## Install Requirements

pip install -r requirements.txt

## Run Training Pipeline


python src/pipeline.py


This generates:

* Trained model
* Model artifacts
* Encoded feature columns


## Run Streamlit Application

streamlit run app.py


# Model Evaluation Metrics

The following regression metrics were used:

* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* Mean Absolute Error (MAE)
* Mean Absolute Percentage Error (MAPE)


# Model Highlights

* End-to-end regression pipeline
* Outlier handling using Winsorization
* Multiple model experimentation
* Feature importance analysis
* Real-time fare prediction
* Streamlit deployment
* Docker containerization
* Logging integration
* Model serialization using Pickle


# Future Improvements

* Hyperparameter tuning
* CI/CD integration
* Cloud deployment
* Real-time API integration
* Advanced feature engineering

# Author

Raj Kiran Reddy