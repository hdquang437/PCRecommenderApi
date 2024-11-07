import pandas as pd
from .model import WideAndDeepModel
from .data_processing import update_main_data

def train_wide_and_deep_model():
    update_main_data()  # Cập nhật dữ liệu chính

    users = pd.read_csv("train_data/user_data.csv")
    items = pd.read_csv("train_data/item_data.csv")
    interactions = pd.read_csv("train_data/interaction_data.csv")

    X_train = pd.concat([users, items], axis=1)
    y_train = interactions["click_times"]

    model = WideAndDeepModel()
    model.build_model(user_features=users.columns, item_features=items.columns)
    model.train(X_train, y_train, epochs=10)

    return model
