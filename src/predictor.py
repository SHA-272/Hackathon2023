from catboost import CatBoostRegressor

model = CatBoostRegressor()
#model.load_model('model.cbm')

def predict(data):
    model.predict(data)