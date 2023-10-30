from catboost import CatBoostRegressor

model = CatBoostRegressor()
model.load_model('./model.cbm')



def predict(X_data):
    return model.predict(X_data)