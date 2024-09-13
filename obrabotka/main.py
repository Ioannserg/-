import pandas as pd
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
import warnings
warnings.filterwarnings('ignore')
#
cars = pd.read_csv("C:/Users/пользватель/OneDrive/Рабочий стол/eto_ono6.csv")
# cars = cars[~cars['model'].isna()]
# cars = cars.dropna(axis=1, how='all')
# cars.duplicated().value_counts()
# cars = cars.drop('engine_volume', axis=1)
# cars.to_csv('C:/Users/пользватель/OneDrive/Рабочий стол/eto_ono3.csv', index=False, encoding='cp1251')

# # Преобразуем тип признака из object в int
#
# # Заменим пропущенные значения средним значением
# cars['horse_power'] = cars['horse_power'].fillna(int(cars['horse_power'].mean()))
# # Заменяем пропущенные значения модой
# cars['configuration'] = cars['configuration'].fillna(cars['configuration'].mode()[0])
# cars['complectation'] = cars['complectation'].fillna(cars['complectation'].mode()[0])
# cars['body_type'] = cars['body_type'].fillna(cars['body_type'].mode()[0])
# cars['drive_type'] = cars['drive_type'].fillna(cars['drive_type'].mode()[0])
# cars['engine_type'] = cars['engine_type'].fillna(cars['engine_type'].mode()[0])
# cars['transmission'] = cars['transmission'].fillna(cars['transmission'].mode()[0])
#
#
# # cars['price_rub'] = cars['price_rub'].astype(str).apply(lambda s: ''.join([x for x in s if x.isdigit()])).astype('int32')
#
#
# print("Cars with restyling:", sum(cars['generation'].apply(lambda x: 'рестайлинг' in x)))
# cars['restyling'] = cars['generation'].apply(lambda x: 'рестайлинг' in x).map({True: 'Да', False: 'Нет'})
# cars['generation'] = cars['generation'].apply(lambda x: x.replace('рестайлинг ', ''))
# cars['generation'] = cars['generation'].apply(lambda x: x.split()[0])
#
#
# cars['engine_volume'] = cars['configuration'].str.extract(r'(\d+\.\d+)').astype(float)
#
#
# packages = {
#     'Базовая': 'Base', 'Base': 'Base',
#     'SE': 'SE', 'Особая серия': 'SE',
#     'Lux': 'Luxe', 'Люкс': 'Luxe',
#     'Норма': 'Norma', 'Norma': 'Norma',
#     'Sport': 'Sport', 'HSE': 'HSE',
#     'Стандарт': 'Standard', 'Standart': 'Standard',
#     'Комфорт': 'Comfort', 'Confort': 'Comfort',
#     'Comfort': 'Comfort', 'Premium': 'Premium',
#     'Enjoy': 'Enjoy', 'Executive': 'Executive',
#     'Special edition': 'SE', 'Limited Edition': 'LE',
#     'Limited': 'LE', 'Active': 'Active',
#     'Prestige': 'Prestige', 'Invite': 'Invite',
#     'Business': 'Business', 'Trendline': 'Trend'
# }
# def same_package(value):
#     for package in packages:
#         if package in value:
#             return packages.get(package)
#     return value
# cars['complectation'] = cars['complectation'].apply(lambda x: same_package(x))
#
# cars = cars[(cars['price_rub'] < cars.price_rub.quantile(0.99)) & (cars['price_rub'] > cars.price_rub.quantile(0.01))]
# cars = cars[~((cars['km_age'] <= 10000) | (cars['km_age'] >= 900000))]
# cars = cars.drop('configuration', axis=1)
#
#
#
# cars.info()
#
#
#
# cars.to_csv('C:/Users/пользватель/OneDrive/Рабочий стол/eto_ono2.csv', index=False, encoding='cp1251')
#





#
# import numpy as np
# import pandas as pd
# import scipy.stats as st
# import seaborn as sns
# import matplotlib.pyplot as plt
#
# import warnings
# warnings.filterwarnings('ignore')
#
# plt.rcParams['figure.figsize'] = (15, 10)
#
# cars = pd.read_csv('C:/Users/пользватель/OneDrive/Рабочий стол/eto_ono2.csv')
#
# price_rub = cars['price_rub']
# # гистограмма
# plt.hist(price_rub, bins=30, color='blue', edgecolor='black')
# plt.title('Распределение price_rub')
# plt.xlabel('Цена в рублях')
# plt.ylabel('Частота')
#
#
#
# price_rub_log = np.log(price_rub + 1)
#
# # Создаем гистограмму логарифмированных данных
# plt.hist(price_rub_log, bins=30, color='green', edgecolor='black')
# plt.title('Логарифмическое распределение price_rub')
# plt.xlabel('Логарифм цены в рублях')
# plt.ylabel('Частота')
#
# # Показываем график
# plt.show()
#
#
#
#
# #вещественные
# def plot_real_dist():
#     features = ['horse_power', 'year', 'km_age']
#     f, ax = plt.subplots(3, 1, figsize=(15, 12))
#     for idx, feature in enumerate(features):
#         sns.distplot(cars[(cars[feature] < cars[feature].quantile(.99)) \
#                           & (cars[feature] > cars[feature].quantile(.1))][feature], ax=ax[idx], color='k')
#         # ax[idx].set_title(f"Признак {feature}")
#
#     f.tight_layout()
#     f.subplots_adjust(top=0.95)
#     f.suptitle("Распределения вещественных признаков")
#     plt.show()
#
#
# plot_real_dist()
#
#
#
#
#
# #категориальные
# def plot_categorical_dist(feature, figsize=(14,10), vertical=False):
#     f, ax = plt.subplots(figsize=figsize)
#     if vertical:
#         sns.countplot(x=feature, data=cars, order=cars[feature].value_counts().iloc[:20].index, palette="hls")
#     else:
#         sns.countplot(y=feature, data=cars, order=cars[feature].value_counts().iloc[:20].index, palette="hls")
#     ax.set(xlabel="")
#     ax.set(ylabel="")
#     sns.despine(trim=True, left=True)
#     b, t = plt.ylim()
#     plt.ylim(b+0.5, t-0.5)
#     f.tight_layout()
#     plt.title(f"Распределение признака {feature}")
#     plt.show()
# plot_categorical_dist('wheel')
#
#
#
#
#
#
#
#
#
# #категориальные зависимость от цены
# def plot_dependence(feature, data=cars, figsize=(15, 8)):
#     f, ax = plt.subplots(figsize=figsize)
#     ax.set_xscale("log")
#     sns.boxplot(x="price_rub", y=feature, data=data.sort_values('price_rub', ascending=False),
#                 whis=1.5, palette='RdBu_r', showfliers=False)
#     ax.xaxis.grid(True)
#     ax.set(ylabel="")
#     sns.despine(trim=True, left=True)
#     b, t = plt.ylim()
#     plt.ylim(b + 0.5, t - 0.5)
#     f.tight_layout()
#     plt.title(f"Зависимость цены от {feature}")
#     plt.show()
#
#
# def only_common_values(feature, threshold=150):
#     values_to_plot = set()
#
#     for idx, value in enumerate(cars[feature].value_counts().index):
#         total_ads = cars[feature].value_counts()[idx]
#         if total_ads < threshold:
#             break
#         values_to_plot.add(value)
#
#     data = cars[cars[feature].apply(lambda x: x in values_to_plot)]
#     return data
#
#
#
# plot_dependence('restyling', figsize=(15,4))
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# #Вещественные признаки зависимость цены
# cars['log_price'] = cars['price_rub'].apply(lambda x: np.log(x))
#
# # Подготовка списка переменных для анализа
# variables = ['year', 'horse_power', 'km_age', 'owners_count', 'log_price', 'engine_volume']
#
# # Сначала рисуем тепловую карту корреляции
# plt.figure(figsize=(15,10))
# corr_matrix = cars[variables].corr()
# sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="YlGnBu", annot_kws={"size": 13})
# plt.ylim(plt.ylim()[0]+0.5, plt.ylim()[1]-0.5)  # Корректировка пределов для корректного отображения
# plt.tight_layout()
# plt.show()
#
# # Затем создаем парные графики
# sns.pairplot(cars[variables])
# plt.show()



import pandas as pd
import matplotlib.pyplot as plt

# Подсчет количества автомобилей по марке
mark_counts = cars['mark'].value_counts()
unique_marks_count = cars['mark'].nunique()
print(f'Количество различных марок в выборке: {unique_marks_count}')

# Определение порога и создание новой категории для малых значений
threshold = 0.02 * mark_counts.sum()  # 2.2% от общего количества
small_counts = mark_counts[mark_counts < threshold]
other_count = small_counts.sum()

# Создание новой серии данных с объединенной категорией "Другие"
mark_counts = mark_counts[mark_counts >= threshold]
mark_counts['Другие'] = other_count

# Построение круговой диаграммы с уникальными цветами
colors = plt.get_cmap('tab20').colors  # Используем палитру tab20
plt.figure(figsize=(10, 10))
mark_counts.plot(kind='pie', autopct='%1.1f%%', colors=colors[:len(mark_counts)])
plt.title('Распределение автомобилей по признаку "Mark"')
plt.ylabel('')  # Скрыть метку оси y
plt.show()
unique_marks_count = cars['mark'].nunique()
print(f'Количество различных марок в выборке: {unique_marks_count}')