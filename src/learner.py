import time
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

model = CatBoostRegressor(iterations=1000, task_type='GPU', devices='0:1', learning_rate=1, depth=6)

# Определите целевую переменную и признаки
target = 'number_of_crimes_in_the_field_of_information_security'
features = [
    'male_population',
    'female_population',
    'number_of_unemployed_population',
    'population_mortality_rate',
    'average_labor_payment',
    'crime_IT_rate',
    'users_attack',
    'inflation_rate',
    'standard_of_living',
    'population_digitization_level',
    'number_of_companies_in_the_information_technology_sector',
    'number_of_population_educated_in_IT',
    'number_of_known_hacking_communities',
    'number_of_investments_in_cybersecurity_sector'
]



def learn(X_train, y_train):
    # Обучите модель CatBoost с пуассоновской регрессией
    print("Start learning...")
    model.fit(X_train, y_train)
    print("End learning")



def save():
    print("Save the model...")
    if model: model.save_model(f'model.cbm')
    print("Model saved")



if __name__ == '__main__':
    data = pd.read_csv(input('Enter path:'), delimiter=';')

    X = data[features]
    y = data[target]
    print(X.values)

    # Разделите данные на обучающий и тестовый наборы
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    learn(X_train, y_train)

    # Сделайте предсказания на тестовом наборе
    y_pred = model.predict(X_test)
    print('Prdicted', y_pred)
    print('Real', y_test.values)

    # Оцените качество модели
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    if input("Save? (y/n)") == 'y': save()