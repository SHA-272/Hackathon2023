from catboost import CatBoostRegressor
import glob, os

# model = CatBoostRegressor()
# model_files = glob.glob('./models/')

# if len(model_files) <= 0: exit("No models, use learner.py")

# latest_model_file = max(model_files, key=os.path.getctime)
# model.load_model(latest_model_file)



def predict(*X_data):
    return X_data
    return model.predict(X_data)