# Lifespan Prediction of Industrial Materials

This project focuses on predicting the lifespan (in hours) of materials used in industrial components based on simulated data. The dataset includes various features related to material composition, manufacturing processes, and structural defects. The project explores regression techniques, feature engineering, and material science applications.

## Problem Statement

Given a dataset with multiple features related to material properties, the objective is to predict the lifespan of the materials accurately. The solution involves end-to-end data processing, model experimentation, and deployment.

---

## Project Features

### 1. Exploratory Data Analysis (EDA)
- Comprehensive analysis to understand data distribution, patterns, and anomalies.

### 2. Data Pipeline
- **Data Ingestion:** Reading and validating the dataset.
- **Data Transformation:** Cleaning, scaling, and feature engineering for improved predictions.

### 3. Machine Learning Models
We tested various regression models, tuned hyperparameters, and selected the best-performing model.

#### ML Models Evaluated:
- **RandomForestRegressor**
- **DecisionTreeRegressor**
- **GradientBoostingRegressor**
- **LinearRegression**
- **XGBRegressor** *(Best performing: 96% accuracy)*
- **AdaBoostRegressor**
- **Ridge**
- **Lasso**

#### Hyperparameter Tuning
Models were fine-tuned using grid search for optimal performance. Below are the hyperparameters used:

```json
{
    "Decision Tree": {
        "criterion": [
            "squared_error",
            "friedman_mse",
            "absolute_error",
            "poisson"
        ]
    },
    "Random Forest": {
        "n_estimators": [8, 16, 32, 64, 128, 256]
    },
    "Gradient Boosting": {
        "learning_rate": [0.1, 0.01, 0.05, 0.001],
        "subsample": [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
        "n_estimators": [8, 16, 32, 64, 128, 256]
    },
    "Linear Regression": {},
    "XGBRegressor": {
        "learning_rate": [0.1, 0.01, 0.05, 0.001],
        "n_estimators": [8, 16, 32, 64, 128, 256]
    },
    "AdaBoost Regressor": {
        "learning_rate": [0.1, 0.01, 0.5, 0.001],
        "n_estimators": [8, 16, 32, 64, 128, 256]
    },
    "Ridge": {},
    "Lasso": {}
}
```
### 4. Best Model Performance
The XGBRegressor achieved the highest accuracy of 96%, making it the preferred model for deployment.

### 5. Deployment
A Flask application was built to host the model and provide predictions through a user-friendly interface.

## Screenshots
When we load the application we are able to see the application like below 
<br/>
<img width="418" alt="s1" src="https://github.com/shivakumar-ravichandran/Material-Lifespan-Prediction/blob/main/screenshots/s1.png">

The prediction will appear in popup window.<br/>
<img width="418" alt="s1" src="https://github.com/shivakumar-ravichandran/Material-Lifespan-Prediction/blob/main/screenshots/s2.png">
<br/>

## How to Use This Repository
### Requirements
- Python 3.8+
- Required libraries are listed in *requirements.txt*.

### Steps to Run
1. Steps to Run
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run the Flask application:
```bash
python app.py
```
4. Access the UI in your browser at http://127.0.0.1:5000/predict.

# Author
This project was created by Shivakumar Ravichandran.
For any questions or support, feel free to reach out via email: shivakumar.mcet@gmail.com.
