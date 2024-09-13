import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import os

# Load data
cars = pd.read_csv('C:/Users/пользватель/OneDrive/Рабочий стол/eto_ono6.csv')
cars = cars.sample(frac=1, random_state=80)  # Shuffle the data

# Create max_price_for_category column and save it
max_prices = cars.groupby('price_category')['price_rub'].max().rename('max_price_for_category')
cars = cars.join(max_prices, on='price_category')
max_price_df = cars[['price_category', 'max_price_for_category']].drop_duplicates()

# Define categorical and numeric features
CATEGORICAL_FEATURES = ['mark', 'model', 'generation', 'body_type', 'engine_type',
                        'transmission', 'color', 'drive_type', 'wheel', 'complectation', 'restyling', 'condition']
NUMERIC_FEATURES = ['horse_power', 'year', 'km_age', 'owners_count', 'engine_volume']

# Preprocess categorical and numeric features
onehot_encoder = OneHotEncoder()
cars_categorical = onehot_encoder.fit_transform(cars[CATEGORICAL_FEATURES]).toarray()  # Convert to dense array manually
scaler = StandardScaler()
cars_numeric = scaler.fit_transform(cars[NUMERIC_FEATURES])
X = np.concatenate([cars_categorical, cars_numeric], axis=1)

# Prepare target variables for category model and price model
y_category = cars['price_category'].values
y_price = np.log(cars['normalized_price'].values)

# Split data into training and testing sets for both models
X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(X, y_category, test_size=0.3, random_state=80)
X_train_price, X_test_price, y_train_price, y_test_price = train_test_split(X, y_price, test_size=0.3, random_state=80)

# Model for category classification
num_categories = cars['price_category'].max() + 1
category_model = Sequential([
    Input(shape=(X_train_cat.shape[1],)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(num_categories, activation='softmax')
])
category_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
category_model.fit(X_train_cat, y_train_cat, validation_split=0.2, epochs=100, callbacks=[EarlyStopping(monitor='val_loss', patience=10)])
category_model.evaluate(X_test_cat, y_test_cat)

# Model for price prediction
price_model = Sequential([
    Input(shape=(X_train_price.shape[1],)),  # Explicit Input layer
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)  # Output layer for regression; no activation function needed
])
price_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
price_model.fit(X_train_price, y_train_price, validation_split=0.2, epochs=100, callbacks=[EarlyStopping(monitor='val_loss', patience=10)])
price_model.evaluate(X_test_price, y_test_price)

# Additional machine learning models
ridge_model = Ridge(alpha=1.0).fit(X_train_price, y_train_price)
lasso_model = Lasso(alpha=0.1).fit(X_train_price, y_train_price)
rf_model = RandomForestRegressor(n_estimators=100).fit(X_train_price, y_train_price)
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100).fit(X_train_price, y_train_price)

# Define and check the directory path, then create if it does not exist
directory_path = 'path_to_my_model'  # Adjust this path as needed
if not os.path.exists(directory_path):
    os.makedirs(directory_path)


# Save all models and preprocessing objects to the directory
# max_price_df.to_csv(f'{directory_path}/max_price_for_category.csv', index=False)
# category_model.save(f'{directory_path}/category_model.keras')
# price_model.save(f'{directory_path}/price_model2.keras')
# joblib.dump(onehot_encoder, f'{directory_path}/onehot_encoder.joblib')
# joblib.dump(scaler, f'{directory_path}/scaler.joblib')
# joblib.dump(ridge_model, f'{directory_path}/ridge_model.joblib')
# joblib.dump(lasso_model, f'{directory_path}/lasso_model.joblib')
# joblib.dump(rf_model, f'{directory_path}/rf_model.joblib')
# joblib.dump(xgb_model, f'{directory_path}/xgb_model.joblib')
