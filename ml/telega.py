import telebot
from telebot import types
from tensorflow.keras.models import load_model
import joblib
import numpy as np
import pandas as pd

bot_token = ...
bot = telebot.TeleBot(bot_token)

# Загрузка моделей и объектов предварительной обработки
category_model = load_model('path_to_my_model/category_model.h5')
price_model = load_model('path_to_my_model/price_model.h5')
ridge_model = joblib.load('path_to_my_model/ridge_model.joblib')
lasso_model = joblib.load('path_to_my_model/lasso_model.joblib')
rf_model = joblib.load('path_to_my_model/rf_model.joblib')
xgb_model = joblib.load('path_to_my_model/xgb_model.joblib')
onehot_encoder = joblib.load('path_to_my_model/onehot_encoder.joblib')
scaler = joblib.load('path_to_my_model/scaler.joblib')

max_price_dict = pd.read_csv('path_to_my_model/max_price_for_category.csv').set_index('price_category')['max_price_for_category'].to_dict()

FEATURES = [
    ('mark', "Введите марку автомобиля"),
    ('model', "Введите модель автомобиля"),
    ('generation', "Введите поколение автомобиля"),
    ('body_type', "Введите тип кузова"),
    ('engine_type', "Введите тип двигателя (Выберите из предложенного)"),
    ('transmission', "Введите тип трансмиссии (Выберите из предложенного)"),
    ('color', "Введите цвет автомобиля"),
    ('drive_type', "Введите тип привода (Выберите из предложенного)"),
    ('wheel', "Введите расположение руля (Выберите из предложенного)"),
    ('complectation', "Введите комплектацию"),
    ('restyling', "Автомобиль рестайлинговый? (Да/Нет)"),
    ('condition', "Введите состояние автомобиля (Выберите из предложенного)"),
    ('horse_power', "Введите мощность двигателя (л.с.)"),
    ('year', "Введите год выпуска"),
    ('km_age', "Введите пробег (км)"),
    ('owners_count', "Введите количество владельцев (Выберите из предложенного)"),
    ('engine_volume', "Введите объем двигателя (л)")
]
user_data = {}

def preprocess_input(user_df, onehot_encoder, scaler):
    user_df_categorical = onehot_encoder.transform(user_df[[f[0] for f in FEATURES[:12]]]).toarray()
    user_df_numeric = scaler.transform(user_df[[f[0] for f in FEATURES[12:]]])
    return np.concatenate([user_df_categorical, user_df_numeric], axis=1)

@bot.message_handler(commands=['start'])
def send_welcome(message):
    welcome_text = (
        "Добро пожаловать в AutoValue AI Predictor, ваш интеллектуальный помощник на рынке подержанных автомобилей! "
        "Используя передовые алгоритмы машинного обучения, наш бот предоставляет вам точные оценки цен на подержанные "
        "автомобили в режиме реального времени. Распрощайтесь с догадками и используйте возможности больших данных "
        "для принятия обоснованных решений, покупаете ли вы, продаете или просто просматриваете веб-страницы. "
        "Просто введите данные о вашем автомобиле, и пусть AutoValue AI Predictor за считанные секунды определит его "
        "истинную рыночную стоимость."
    )
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    btn1 = types.KeyboardButton("Оценить автомобиль")
    btn2 = types.KeyboardButton("Связаться с администратором")
    btn3 = types.KeyboardButton("Информация")
    markup.add(btn1, btn2, btn3)
    bot.send_message(message.chat.id, welcome_text, reply_markup=markup)
def send_main_menu(chat_id):
    welcome_text = (
        "Выберите опцию:"
    )
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    btn1 = types.KeyboardButton("Оценить автомобиль")
    btn2 = types.KeyboardButton("Связаться с администратором")
    btn3 = types.KeyboardButton("Информация")
    markup.add(btn1, btn2, btn3)
    bot.send_message(chat_id, welcome_text, reply_markup=markup)

@bot.message_handler(func=lambda message: message.text == "Связаться с администратором")
def contact_admin(message):
    bot.send_message(message.chat.id, "Если возникли вопросы или предложения о сотрудничестве, пожалуйста, напишите @ioannserg")

@bot.message_handler(func=lambda message: message.text == "Информация")
def show_info(message):
    bot.send_message(message.chat.id, "Данный бот не дает гарантию на точное прогнозированние цены. "
                                      "Средняя ошибка прогноза составляет ≈ 40-60 тысяч рублей. "
                                      "Бот разработан в целях написания и защиты дипломной работы. "
                                      "Если бот вышел из строя, введите /start "
                                      "По всем вопросам обращаться к создателю @ioannserg")

@bot.message_handler(func=lambda message: message.text == "Оценить автомобиль")
def evaluate_car(message):
    ask_feature(message, 0)

def ask_feature(message, index):
    feature_name, feature_prompt = FEATURES[index]
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
    if feature_name == 'engine_type':
        markup.add('Бензин', 'Дизель', 'Электро')
    elif feature_name == 'transmission':
        markup.add('Автомат', 'Механика', 'Робот', 'Вариатор')
    elif feature_name == 'drive_type':
        markup.add('Передний', 'Задний', 'Полный')
    elif feature_name == 'wheel':
        markup.add('Левый', 'Правый')
    elif feature_name == 'restyling':
        markup.add('Да', 'Нет')
    elif feature_name == 'condition':
        markup.add('отличное', 'требует ремонта')
    elif feature_name == 'owners_count':
        markup.add('1', '2', '3', '4')
    msg = bot.send_message(message.chat.id, feature_prompt, reply_markup=markup if markup.keyboard else None)
    bot.register_next_step_handler(msg, process_feature, index)

def process_feature(message, index):
    user_data[FEATURES[index][0]] = message.text
    next_index = index + 1
    if next_index < len(FEATURES):
        ask_feature(message, next_index)
    else:
        predict_price(message.chat.id)
def predict_price(chat_id):
    try:
        # Создание DataFrame из собранных данных пользователя
        user_df = pd.DataFrame([user_data])

        # Предварительная обработка введенных данных
        processed_input = preprocess_input(user_df, onehot_encoder, scaler)

        # Получение категории цены с помощью модели классификации
        predicted_category = category_model.predict(processed_input)
        predicted_category = np.argmax(predicted_category, axis=1)[0]

        # Установка категории цены в DataFrame
        user_df['price_category'] = predicted_category

        # Вычисление максимального коэффициента цены для данной категории
        max_price_factor = max_price_dict.get(predicted_category, 1)

        # Получение предсказаний от различных моделей
        predictions = {
            "TensorFlow": price_model.predict(processed_input),
            "Ridge": ridge_model.predict(processed_input),
            "Lasso": lasso_model.predict(processed_input),
            "Random Forest": rf_model.predict(processed_input),
            "XGBoost": xgb_model.predict(processed_input)
        }

        # Форматирование и отправка результатов предсказания
        results = []
        for name, pred in predictions.items():
            price = np.exp(pred[0]) * max_price_factor
            if isinstance(price, np.ndarray):
                price = price.item()
            results.append(f"Прогноз стоимости в RUB ({name}): {price:.2f}")

        bot.send_message(chat_id, "\n".join(results))
        mark = user_data['mark'].lower()
        if mark == 'ваз (lada)':
            mark = 'vaz_lada'
        else:
            mark = mark
        model = user_data['model'].lower().replace(' ', '_')
        avito_url = f"https://www.avito.ru/naberezhnye_chelny/avtomobili/{mark}/{model}?radius=200&searchRadius=200"
        bot.send_message(chat_id, f"Посмотрите похожие объявления здесь: {avito_url}")
    except Exception as e:
        # Обработка любых исключений, возникших во время предсказания
        bot.send_message(chat_id,
                         f"Произошла ошибка при расчете стоимости: {e}\nНе правильно введенные данные или сбой в системе.")
    finally:
        # Возвращение пользователя в главное меню
        send_main_menu(chat_id)

if __name__ == '__main__':
    bot.polling()
