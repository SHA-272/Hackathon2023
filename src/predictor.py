from catboost import CatBoostRegressor
import glob, os

model = CatBoostRegressor()
model.load_model('./model.cbm')



def predict(X_data):
    return model.predict(X_data)