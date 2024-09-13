import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

RANDOM_STATE = 80
cars = pd.read_csv('C:/Users/пользватель/OneDrive/Рабочий стол/12.csv')

# Перемешиваем данные
cars = cars.sample(frac=1, random_state=RANDOM_STATE)

CATEGORICAL_FEATURES = ['Марка_Модель', 'Тип двигателя', 'Коробка передач', 'Привод', 'Тип кузова', 'Цвет', 'Поколение']
NUMERIC_FEATURES = ['Год выпуска', 'Пробег', 'Владельцев по ПТС', 'Объём двигателя', 'Мощность', 'рестайлинг']

# Разделяем данные на признаки и целевую переменную
X = cars.drop(['Цена'], axis=1)
y = np.log(cars['Цена'])

# Разделение на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE)

# Создание трансформера для преобразования категориальных признаков
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), NUMERIC_FEATURES),
        ('cat', OneHotEncoder(), CATEGORICAL_FEATURES)
    ],
    remainder='passthrough'
)

# Преобразование данных
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# Создание и компиляция модели нейронной сети
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Обучение модели
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=0)

# Оценка модели
y_preds = model.predict(X_test)
y_preds = np.squeeze(y_preds)  # Преобразование формы предсказаний

print('MSE: ', mean_squared_error(y_test, y_preds))
print('R2: ', r2_score(y_test, y_preds))
print('MAE: ', mean_absolute_error(np.exp(y_test), np.exp(y_preds)))
