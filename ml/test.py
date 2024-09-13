import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelBinarizer, StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.linear_model import Ridge, SGDRegressor, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

import warnings
warnings.filterwarnings('ignore')
RANDOM_STATE = 80

# Load data
cars = pd.read_csv('C:/Users/пользватель/OneDrive/Рабочий стол/eto_ono6.csv')

# Creating max_price_for_category column
max_prices = cars.groupby('price_category')['price_rub'].max().rename('max_price_for_category')
cars = cars.join(max_prices, on='price_category')

# Shuffle data
cars = cars.sample(frac=1, random_state=RANDOM_STATE)

CATEGORICAL_FEATURES = ['mark', 'model', 'generation', 'body_type', 'engine_type',
                        'transmission', 'color', 'drive_type', 'wheel', 'complectation', 'price_category']

NUMERIC_FEATURES = ['horse_power', 'year', 'km_age', 'owners_count', 'engine_volume']

def preprocess_features(dataframe):
    lb = LabelBinarizer()
    dataframe['condition'] = lb.fit_transform(dataframe['condition'])
    dataframe['restyling'] = lb.fit_transform(dataframe['restyling'])

    dataframe = pd.get_dummies(dataframe, columns=CATEGORICAL_FEATURES)
    print(f"We have {dataframe.shape[1]} features after one-hot encoding")

    scaler = StandardScaler()
    poly = PolynomialFeatures(interaction_only=True)

    numeric_data = scaler.fit_transform(dataframe[NUMERIC_FEATURES])
    numeric_data_poly = poly.fit_transform(numeric_data)
    poly_features_names = [f"poly_{i}" for i in range(numeric_data_poly.shape[1])]
    dataframe = dataframe.drop(NUMERIC_FEATURES, axis=1)
    dataframe[poly_features_names] = numeric_data_poly

    return dataframe

def make_prediction(model, X_train, X_test, y_train, y_test, params_fixed, params_grid=None):
    if params_grid:
        estimator = GridSearchCV(model(**params_fixed), params_grid, cv=KFold(n_splits=3), verbose=True)
    else:
        estimator = model(**params_fixed)

    estimator.fit(X_train, y_train)
    y_preds_train = estimator.predict(X_train)
    y_preds_test = estimator.predict(X_test)

    # Denormalize predictions
    y_preds_train_unnormalized = np.exp(y_preds_train) * X_train['max_price_for_category']
    y_preds_test_unnormalized = np.exp(y_preds_test) * X_test['max_price_for_category']
    y_train_unnormalized = np.exp(y_train) * X_train['max_price_for_category']
    y_test_unnormalized = np.exp(y_test) * X_test['max_price_for_category']

    # Recalculate metrics for denormalized values
    mse_train = mean_squared_error(y_train_unnormalized, y_preds_train_unnormalized)
    r2_train = r2_score(y_train_unnormalized, y_preds_train_unnormalized)
    mae_train = mean_absolute_error(y_train_unnormalized, y_preds_train_unnormalized)

    mse_test = mean_squared_error(y_test_unnormalized, y_preds_test_unnormalized)
    r2_test = r2_score(y_test_unnormalized, y_preds_test_unnormalized)
    mae_test = mean_absolute_error(y_test_unnormalized, y_preds_test_unnormalized)

    print(f'Train MSE: {mse_train}, Train R2: {r2_train}, Train MAE: {mae_train}')
    print(f'Test MSE: {mse_test}, Test R2: {r2_test}, Test MAE: {mae_test}')

    if params_grid:
        print("Best parameters:", estimator.best_params_)

    return estimator, y_train_unnormalized, y_preds_train_unnormalized, y_test_unnormalized, y_preds_test_unnormalized

# Preprocessing data
cars = preprocess_features(cars)

# Prepare data for training
y = np.log(cars['normalized_price'])  # Log-normalizing price
X = cars.drop(['price_rub', 'normalized_price'], axis=1)

# Splitting data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE, shuffle=True)

# Train Lasso model with Grid Search
lasso_params_fixed = {'random_state': RANDOM_STATE}
lasso_params_grid = {'alpha': np.linspace(1e-05, 1e-03, num=10)}
print("\tLasso model with Grid Search\n")
lasso_model, lasso_y_train, lasso_y_preds_train, lasso_y_test, lasso_y_preds_test = make_prediction(Lasso, X_train, X_test, y_train, y_test, lasso_params_fixed, lasso_params_grid)

# Train Ridge model
print("\tRidge model (Baseline)\n")
ridge_params = {'random_state': RANDOM_STATE}
ridge_model, ridge_y_train, ridge_y_preds_train, ridge_y_test, ridge_y_preds_test = make_prediction(Ridge, X_train, X_test, y_train, y_test, ridge_params)

# Train Random Forest
print("\tRandom Forest (Baseline)\n")
rf_params = {'random_state': RANDOM_STATE}
rf_model, rf_y_train, rf_y_preds_train, rf_y_test, rf_y_preds_test = make_prediction(RandomForestRegressor, X_train, X_test, y_train, y_test, rf_params)

# Train XGBoost model
print("\tXGBoost model (Baseline)\n")
xgb_params_fixed = {'random_state': RANDOM_STATE, 'use_label_encoder': False, 'eval_metric': 'rmse'}
xgb_params_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2]
}
xgb_model, xgb_y_train, xgb_y_preds_train, xgb_y_test, xgb_y_preds_test = make_prediction(xgb.XGBRegressor, X_train, X_test, y_train, y_test, xgb_params_fixed, xgb_params_grid)







