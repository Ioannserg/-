from selenium import webdriver
from selenium.common import NoSuchElementException
from selenium_stealth import stealth
import time
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import csv
import os
import random


#создаем csv
desktop_path = os.path.expanduser("~\\OneDrive\\Рабочий стол")
csv_file_path = os.path.join(desktop_path, "avito_data16_mersedes.csv")
csv_headers = ["Marka", "Model", "Год выпуска", "Поколение", "Пробег", "История пробега", "ПТС", "Владельцев по ПТС", "Состояние", "Модификация", "Объём двигателя", "Тип двигателя", "Коробка передач", "Привод", "Комплектация", "Тип кузова", "Цвет", "Руль", "VIN или номер кузова", "Обмен"]


with open(csv_file_path, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(csv_headers)


    # не палимся сайту
    service = webdriver.ChromeOptions()
    service.add_argument("start-maximized")
    service.add_experimental_option("excludeSwitches", ["enable-automation"])
    service.add_experimental_option('useAutomationExtension', False)

    service = Service('C:\\Users\\пользватель\\PycharmProjects\\pythonProject\\drivers\\chromedriver.exe')
    driver = webdriver.Chrome(service=service)

    stealth(driver,
            languages=["ru-RU", "ru"],
            vendor="Google Inc.",
            platform="Win64",
            webgl_vendor="Intel Inc.",
            renderer="Intel Iris OpenGL Engine",
            fix_hairline=True,
            )

    data = set()

    # парсим ссылки
    for page in range(1, 14):
        url = f"https://www.avito.ru/naberezhnye_chelny/avtomobili/mercedes-benz/levyy_rul-ASgBAgICAkTwCqyKAeC2DeiYKA?cd=1&f=ASgBAgICBUTwCqyKAfIKsIoBhhTI5gHgtg3omCj68A_ou_cC&p={page}&radius=200&searchRadius=200"
        driver.get(url)
        time.sleep(10+random.random())

        blocks = driver.find_element(By.CSS_SELECTOR, "#app > div > div.styles-singlePageWrapper-eKDyt > div > div.index-center-_TsYY.index-center_withTitle-_S7ge.index-center_noMarginTop-xAh5X.index-centerWide-_7ZZ_.index-center_marginTop_1-ewXHO > div.index-inner-dqBR5.index-innerCatalog-ujLwf > div.index-content-_KxNP > div.index-root-KVurS > div.items-items-kAJAg")
        posts = blocks.find_elements(By.CLASS_NAME, "styles-module-theme-CRreZ")

        for post in posts:
            try:
                title = post.find_element(By.CLASS_NAME, "iva-item-slider-pYwHo").find_element(By.TAG_NAME,
                                                                                               "a").get_attribute(
                    "href")

                data.add(title)



            except NoSuchElementException:

                continue

    data_list = list(data)


#Собираем нужные данные
    for url in data_list:
        driver.get(url)
        time.sleep(17+random.random()+random.random()+random.random())
        try:
            marka = driver.find_element(By.CSS_SELECTOR,
                                        "#app > div > div.index-root-k1Ib4.index-responsive-aOpFS.index-page_default-_b5bD > div:nth-child(1) > div > div.style-item-view-PCYlM > div.style-item-navigation-In5Jr > div:nth-child(2) > span:nth-child(5) > a > span").text
            model = driver.find_element(By.CSS_SELECTOR,
                                         "#app > div > div.index-root-k1Ib4.index-responsive-aOpFS.index-page_default-_b5bD > div:nth-child(1) > div > div.style-item-view-PCYlM > div.style-item-navigation-In5Jr > div:nth-child(2) > span:nth-child(6) > a > span").text
            params_elements = driver.find_elements(By.CLASS_NAME, 'params-paramsList__item-appQw')
            price = driver.find_element(By.CSS_SELECTOR,
                                       "#app > div > div.index-root-k1Ib4.index-responsive-aOpFS.index-page_default-_b5bD > div:nth-child(1) > div > div.style-item-view-PCYlM > div.style-item-view-content-SDgKX > div.style-item-view-content-right-rxJqW > div.style-item-view-info-HCcXB > div > div > div.style-item-view-price-block-WSyYk > div > div.styles-module-theme-CRreZ > div > div:nth-child(1) > div > span > span > span:nth-child(1)").text

            params_data = {
                'Год выпуска': None,
                'Поколение': None,
                'Пробег': None,
                'История пробега': None,
                'ПТС': None,
                'Владельцев по ПТС': None,
                'Состояние': None,
                'Модификация': None,
                'Объём двигателя': None,
                'Тип двигателя': None,
                'Коробка передач': None,
                'Привод': None,
                'Комплектация': None,
                'Тип кузова': None,
                'Цвет': None,
                'Руль': None,
                'VIN или номер кузова': None,
                'Обмен': None
            }

            for param_element in params_elements:
                param_parts = param_element.text.split(':')
                if len(param_parts) == 2:
                    param_name = param_parts[0].strip()
                    param_value = param_parts[1].strip()
                    if param_name in params_data:
                        params_data[param_name] = param_value

            params_values = list(params_data.values())

            writer.writerow([marka, model] + params_values + [price])
        except NoSuchElementException:

            continue

    driver.quit()
