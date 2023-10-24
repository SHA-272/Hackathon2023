import time
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

model = CatBoostRegressor(iterations=1000, depth=6, loss_function='Poisson', verbose=200)

# Определите целевую переменную и признаки
target = 'количество_преступлений_в_сфере_информационной_безопасности'
features = [
    'численность_мужского_населения',
    'численность_женского_населения',
    'количество_безработного_населения',
    'процент_раскрываемости_преступлений',
    'численность_сотрудников_правоохранительных_органов',
    'смертность_населения',
    'количество_ранее_совершенных_преступлений_в_сфере',
    'средний_размер_оплаты_труда',
    'уровень_инфляции',
    'уровень_жизни_населения',
    'уровень_цифровизации_населения',
    'количество_компаний_в_сфере_информационных_технологий',
    'количество_населения_получившего_образование_в_ИТ',
    'количество_известных_хакерских_сообществ',
    'количество_инцидентов_кибербезопасности',
    'количество_инвестиций_в_сферу_кибербезопасности'
]



def learn(X_train, y_train):
    # Обучите модель CatBoost с пуассоновской регрессией
    print("Start learning...")
    model.fit(X_train, y_train)
    print("End learning")



def save():
    print("Save the model...")
    if model: model.save_model(f'model_{time.time()}.cbm')
    print("Model saved")



if __name__ == '__main__':
    data = pd.read_csv('train_data.csv')

    X = data[features]
    y = data[target]

    # Разделите данные на обучающий и тестовый наборы
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    learn(X_train, y_train)

    # Сделайте предсказания на тестовом наборе
    y_pred = model.predict(X_test)
    print(y_pred)

    # Оцените качество модели
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    if input("Save? (y/n)") == 'y': save()