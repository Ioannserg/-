import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer, StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.linear_model import Ridge, Lasso, ElasticNet, SGDRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

RANDOM_STATE = 80
cars = pd.read_csv('C:/Users/пользватель/OneDrive/Рабочий стол/eto_ono3.csv')

# Перемешаем данные
cars = cars.sample(frac=1, random_state=RANDOM_STATE)

CATEGORICAL_FEATURES = ['mark', 'model', 'generation', 'color', 'engine_type', 'body_type', 'transmission', 'drive_type', 'complectation']
NUMERIC_FEATURES = ['year', 'km_age', 'owners_count', 'horse_power']

def preprocess_categorical_features(dataframe):
    # Преобразование бинарных признаков
    lb = LabelBinarizer()
    dataframe['condition'] = lb.fit_transform(dataframe['condition'])
    dataframe['restyling'] = lb.fit_transform(dataframe['restyling'])
    dataframe['wheel'] = lb.fit_transform(dataframe['wheel'])

    categorical_columns = dataframe.select_dtypes(include=['object']).columns.tolist()
    dataframe_encoded = pd.get_dummies(dataframe, columns=categorical_columns)
    dataframe_encoded.columns = dataframe_encoded.columns.astype(str)
    print(f"We have {dataframe_encoded.shape[1]} features after one-hot encoding")
    return dataframe_encoded

cars = preprocess_categorical_features(cars)
cars.columns = cars.columns.astype(str)  # Преобразуйте все имена столбцов в строковый тип


def preprocess_numeric_features(X_train, X_test):
    scaler = StandardScaler()
    poly = PolynomialFeatures(interaction_only=True)

    X_train[NUMERIC_FEATURES] = scaler.fit_transform(X_train[NUMERIC_FEATURES])
    X_test[NUMERIC_FEATURES] = scaler.transform(X_test[NUMERIC_FEATURES])

    train_matrix = pd.DataFrame(poly.fit_transform(X_train[NUMERIC_FEATURES]), index=X_train.index)
    test_matrix = pd.DataFrame(poly.transform(X_test[NUMERIC_FEATURES]), index=X_test.index)

    return X_train.join(train_matrix), X_test.join(test_matrix)

def make_grid_cv(estimator, params_grid):
    cv = KFold(n_splits=3)
    grid_cv = GridSearchCV(estimator, params_grid, cv=cv, verbose=True)
    return grid_cv

def make_prediction(model, params_fixed, params_grid=None):
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(cars.drop(['price_rub'], axis=1),
                                                        np.log(cars['price_rub']), test_size=0.3,
                                                        shuffle=True, random_state=RANDOM_STATE)

    # Standardize data for linear models
    if model in [Ridge, Lasso, ElasticNet, SGDRegressor]:
        X_train, X_test = preprocess_numeric_features(X_train, X_test)

    X_train.columns = X_train.columns.astype(str)  # Преобразуем названия столбцов в строковый тип
    X_test.columns = X_test.columns.astype(str)  # Преобразуем названия столбцов в строковый тип

    estimator = model(**params_fixed)

    if params_grid:
        estimator = make_grid_cv(estimator, params_grid)

    if model in [xgb.XGBRegressor]:
        X_train.columns = [str(col).replace('[', '_').replace(']', '_').replace('<', '_') for col in X_train.columns]
        X_test.columns = [str(col).replace('[', '_').replace(']', '_').replace('<', '_') for col in X_test.columns]

    estimator.fit(X_train, y_train)
    y_preds = estimator.predict(X_test)
    print('MSE: ', mean_squared_error(y_test, y_preds))
    print('R2: ', r2_score(y_test, y_preds))
    print('MAE: ', mean_absolute_error(np.exp(y_test), np.exp(y_preds)))
    return estimator, y_test, y_preds



cars = preprocess_categorical_features(cars)
# fixed_params = {'random_state': RANDOM_STATE}
# print("\tRidge model (with new feature)\n")
# make_prediction(Ridge, fixed_params)
#
#
# print("\tRandom Forest\n")
# rf, _, _ = make_prediction(RandomForestRegressor, fixed_params)
#
# fixed_params_lasso_elastic = {'random_state': RANDOM_STATE, 'max_iter': 10000}
# print("\tLasso model (GridSearch)\n")
# params_grid = {'alpha': np.linspace(1e-05, 1e-03, num=10)}
# lr, _, _ = make_prediction(Lasso, fixed_params_lasso_elastic, params_grid)
# print(lr.best_params_)
#
#
# print("\tRidge model (GridSearch)\n")
# params_grid = {'alpha': np.linspace(0.9, 2, num=10)}
# lr, _, _ = make_prediction(Ridge, fixed_params, params_grid)
# print(lr.best_params_)
#
#
# print("\tElasticNet model (GridSearch)\n")
# params_grid = {'alpha': np.linspace(1e-05, 1e-04, num=10)}
# lr, _, _ = make_prediction(ElasticNet, fixed_params_lasso_elastic, params_grid)
# print(lr.best_params_)


print("\tXGBregressor\n")
fixed_params = {'random_state': RANDOM_STATE, 'n_estimators': 6000,
                'gamma': 0, 'learning_rate': 0.03,
                'max_depth': 4,
                'min_child_weight': 2,
                'objective': 'reg:squarederror',
                'reg_alpha': 0.75, 'reg_lambda': 0.5,
                'subsample': 0.8}
estimator, y_true, y_preds = make_prediction(xgb.XGBRegressor, fixed_params)