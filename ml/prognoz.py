from tensorflow.keras.models import load_model
import joblib
import numpy as np
import pandas as pd

# Paths to your local machine learning models and preprocessing objects
category_model_path = 'path_to_my_model/category_model.h5'
price_model_path = 'path_to_my_model/price_model.h5'
ridge_model_path = 'path_to_my_model/ridge_model.joblib'
lasso_model_path = 'path_to_my_model/lasso_model.joblib'
rf_model_path = 'path_to_my_model/rf_model.joblib'
xgb_model_path = 'path_to_my_model/xgb_model.joblib'
onehot_encoder_path = 'path_to_my_model/onehot_encoder.joblib'
scaler_path = 'path_to_my_model/scaler.joblib'

# Loading models and preprocessing objects
category_model = load_model(category_model_path)
price_model = load_model(price_model_path)
ridge_model = joblib.load(ridge_model_path)
lasso_model = joblib.load(lasso_model_path)
rf_model = joblib.load(rf_model_path)
xgb_model = joblib.load(xgb_model_path)
onehot_encoder = joblib.load(onehot_encoder_path)
scaler = joblib.load(scaler_path)

# Define categorical and numeric features
CATEGORICAL_FEATURES = ['mark', 'model', 'generation', 'body_type', 'engine_type',
                        'transmission', 'color', 'drive_type', 'wheel', 'complectation', 'restyling', 'condition']
NUMERIC_FEATURES = ['horse_power', 'year', 'km_age', 'owners_count', 'engine_volume']

# Function to request user input
def get_user_input():
    user_input = {}
    print("Please enter car details:")
    for feature in CATEGORICAL_FEATURES:
        user_input[feature] = input(f"{feature.title()}: ")
    for feature in NUMERIC_FEATURES:
        user_input[feature] = float(input(f"{feature.title()}: "))
    return pd.DataFrame([user_input])

# Preprocessing user input
def preprocess_input(user_df, onehot_encoder, scaler):
    user_df_categorical = onehot_encoder.transform(user_df[CATEGORICAL_FEATURES]).toarray()
    user_df_numeric = scaler.transform(user_df[NUMERIC_FEATURES])
    return np.concatenate([user_df_categorical, user_df_numeric], axis=1)

# Loading maximum prices for each category
max_price_path = 'path_to_my_model/max_price_for_category.csv'
max_price_dict = pd.read_csv(max_price_path).set_index('price_category')['max_price_for_category'].to_dict()

# Getting data from user
user_df = get_user_input()

# Preprocessing and predicting price category
processed_input = preprocess_input(user_df, onehot_encoder, scaler)
predicted_category = category_model.predict(processed_input)
predicted_category = np.argmax(predicted_category, axis=1)[0]
print(f"Predicted price category: {predicted_category}")

# Adding predicted category and max price for category to DataFrame
user_df['price_category'] = predicted_category
user_df['max_price_for_category'] = max_price_dict[predicted_category]

# Predicting prices with different models
predicted_price_tf = price_model.predict(processed_input)
predicted_price_ridge = ridge_model.predict(processed_input)
predicted_price_lasso = lasso_model.predict(processed_input)
predicted_price_rf = rf_model.predict(processed_input)
predicted_price_xgb = xgb_model.predict(processed_input)

# Denormalizing prices
max_price_factor = user_df['max_price_for_category'].values[0]
predicted_price_tf = np.exp(predicted_price_tf) * max_price_factor
predicted_price_ridge = np.exp(predicted_price_ridge) * max_price_factor
predicted_price_lasso = np.exp(predicted_price_lasso) * max_price_factor
predicted_price_rf = np.exp(predicted_price_rf) * max_price_factor
predicted_price_xgb = np.exp(predicted_price_xgb) * max_price_factor

# Printing predictions
print(f"Predicted price in RUB (TensorFlow): {predicted_price_tf[0][0]}")
print(f"Predicted price in RUB (Ridge): {predicted_price_ridge[0]}")
print(f"Predicted price in RUB (Lasso): {predicted_price_lasso[0]}")
print(f"Predicted price in RUB (Random Forest): {predicted_price_rf[0]}")
print(f"Predicted price in RUB (XGBoost): {predicted_price_xgb[0]}")
