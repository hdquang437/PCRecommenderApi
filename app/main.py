from fastapi import FastAPI
import pandas as pd
from .train_model import train_wide_and_deep_model
from .data_processing import load_cache_data, update_main_data

app = FastAPI()
model = None

@app.post("/train")
def train_model():
    global model
    model = train_wide_and_deep_model()
    return {"status": "Model trained"}

@app.post("/recommend")
def recommend(user_id: int):
    if model is None:
        return {"status": "No trained model available", "recommendations": []}

    # Lấy dữ liệu chính và kiểm tra trạng thái xóa của các item
    items = pd.read_csv("train_data/item_data.csv")
    deleted_items = items[items["isDelete"] == True]["itemid"]
    items = items[~items["itemid"].isin(deleted_items)]

    # Lọc và tạo dự đoán
    user_data = pd.read_csv("train_data/user_data.csv")
    user = user_data[user_data["userid"] == user_id]
    predictions = model.predict([user, items])

    # Chọn top 10 sản phẩm
    items["prediction"] = predictions
    recommendations = items.nlargest(10, "prediction")["itemid"].tolist()

    return {"user_id": user_id, "recommendations": recommendations}

@app.post("/add_user")
def add_user(user_id: int, age_range: str):
    df = pd.read_csv("train_data/cache_user.csv")
    df = df.append({"userid": user_id, "age_range": age_range}, ignore_index=True)
    df.to_csv("train_data/cache_user.csv", index=False)
    return {"status": "User added to cache"}

@app.post("/add_item")
def add_item(item_id: int, item_type: str, price_range: str):
    df = pd.read_csv("train_data/cache_item.csv")
    df = df.append({"itemid": item_id, "item_type": item_type, "price_range": price_range}, ignore_index=True)
    df.to_csv("train_data/cache_item.csv", index=False)
    return {"status": "Item added to cache"}

@app.post("/add_interaction")
def add_interaction(user_id: int, item_id: int, click_times: int, rating: int, purchase_times: int, is_favorite: bool):
    df = pd.read_csv("train_data/cache_interaction.csv")
    df = df.append({
        "userid": user_id,
        "itemid": item_id,
        "click_times": click_times,
        "rating": rating,
        "purchase_times": purchase_times,
        "is_favorite": is_favorite
    }, ignore_index=True)
    df.to_csv("train_data/cache_interaction.csv", index=False)
    return {"status": "Interaction added to cache"}

@app.post("/delete_item")
def delete_item(item_id: int):
    df = pd.read_csv("train_data/cache_delete_item.csv")
    df = df.append({"itemid": item_id}, ignore_index=True)
    df.to_csv("train_data/cache_delete_item.csv", index=False)
    return {"status": "Item marked for deletion in cache"}
