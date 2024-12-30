import threading
from fastapi import FastAPI
import numpy as np
import pandas as pd
import tensorflow as tf
import os

from app.train_model import periodic_model_rebuild, train_and_save_model
from .data_processing import load_main_data
from .paths import MODEL_PATH, CACHE_DELETED_ITEM_PATH, CACHE_USER_PATH, CACHE_ITEM_PATH, CACHE_INTERACTION_PATH

app = FastAPI()

# Hàm để load mô hình TensorFlow
def load_model():
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH)
    return None

@app.post("/recommend/")
async def recommend(user_id: str, amount: int):
    # Kiểm tra nếu model đã tồn tại
    if not os.path.exists(MODEL_PATH):
        return {"message": "Model has not trained yet."}

    # Tải mô hình đã lưu
    model = tf.keras.models.load_model(MODEL_PATH)

    # Tải dữ liệu
    user_data, item_data, interaction_data = load_main_data()
    cache_deleted_item = pd.read_csv(CACHE_DELETED_ITEM_PATH)

    # Kiểm tra xem user_id có tồn tại trong dữ liệu không
    if user_id not in user_data["userid"].values:
        # nếu user_id không tồn tại, trả về các sản phẩm phổ biến nhất
        return get_popular_items(item_data, interaction_data, amount)

    # Lấy thông tin user
    user_features = user_data[user_data["userid"] == user_id].drop(columns=["userid"]).values

    # Lấy tất cả item hợp lệ (item không bị xóa trong cả item_data và cache_deleted_item)
    # Đọc thông tin các item trong cache_deleted_item
    deleted_item_ids = cache_deleted_item["itemid"].values

    # Lọc ra các item còn tồn tại trong item_data (chưa bị xóa và không có trong cache_deleted_item)
    valid_items = item_data[~item_data["itemid"].isin(deleted_item_ids)]
    
    # Nếu không có item nào hợp lệ để recommend
    if valid_items.empty:
        return {"message": "No valid items are available for recommendation."}

    item_ids = valid_items["itemid"].values
    item_features = valid_items.drop(columns=["itemid", "isDelete"]).values

    # Chuẩn bị đầu vào cho mô hình
    user_ids_input = np.array([user_id] * len(item_ids))  # Nhân bản user_id cho tất cả các item
    interaction_features_input = np.zeros((len(item_ids), 4))  # Mặc định 4 đặc trưng tương tác ban đầu (click_times, rating, purchase_times, is_favorite) là 0

    # Tải mô hình
    model = tf.keras.models.load_model(MODEL_PATH)

    # Dự đoán điểm số
    predictions = model.predict({
        "user_id": user_ids_input,
        "item_id": item_ids,
        "interaction_features": interaction_features_input,
        "user_features": np.repeat(user_features, len(item_ids), axis=0),
        "item_features": item_features,
    })

    # Chọn top "amount" item có điểm số cao nhất
    top_indices = np.argsort(predictions.flatten())[::-1][:amount]
    recommended_items = valid_items.iloc[top_indices]

    # Trả về danh sách item được gợi ý
    return {
        "recommended_items": recommended_items["itemid"].tolist()
    }

@app.post("/add_user")
async def add_user(user_id: int, age_range: str, gender: str):
    df = pd.read_csv(CACHE_USER_PATH)
    df = df.append({"userid": user_id, "age_range": age_range, "gender": gender}, ignore_index=True)
    df.to_csv(CACHE_USER_PATH, index=False)
    return {"status": "User added to cache"}

@app.post("/add_item")
async def add_item(item_id: int, item_type: str, price_range: str):
    df = pd.read_csv(CACHE_ITEM_PATH)
    df = df.append({"itemid": item_id, "item_type": item_type, "price_range": price_range, "is_delete": 0}, ignore_index=True)
    df.to_csv(CACHE_ITEM_PATH, index=False)
    return {"status": "Item added to cache"}

@app.post("/add_interaction")
async def add_interaction(user_id: int, item_id: int, click_times: int, rating: int, purchase_times: int, is_favorite: int):
    df = pd.read_csv(CACHE_INTERACTION_PATH)
    df = df.append({
        "userid": user_id,
        "itemid": item_id,
        "click_times": click_times,
        "rating": rating,
        "purchase_times": purchase_times,
        "is_favorite": is_favorite
    }, ignore_index=True)
    df.to_csv(CACHE_INTERACTION_PATH, index=False)
    return {"status": "Interaction added to cache"}

@app.post("/delete_item")
async def delete_item(item_id: int):
    df = pd.read_csv(CACHE_DELETED_ITEM_PATH)
    df = df.append({"itemid": item_id}, ignore_index=True)
    df.to_csv(CACHE_DELETED_ITEM_PATH, index=False)
    return {"status": "Item marked for deletion in cache"}

# Lấy Item phổ biến
async def get_popular_items(item_data, interaction_data, amount):
    # Tổng hợp dữ liệu để tính toán mức độ phổ biến
    interaction_data['popularity_score'] = (
        interaction_data['click_times'] * 0.5 +  # Đặt clicks weight là 50%
        interaction_data['purchase_times'] * 1.0  # Đặt purchases weight là 100%
    )

    # Tính tổng điểm cho mỗi item
    item_popularity = interaction_data.groupby('itemid')['popularity_score'].sum().reset_index()

    # Kết hợp với item_data để có thêm thông tin chi tiết về mục
    item_popularity = item_popularity.merge(item_data, on='itemid', how='left')

    # Sắp xếp các item theo điểm phổ biến theo thứ tự giảm dần
    top_items = item_popularity.sort_values(by='popularity_score', ascending=False).head(amount)

    # Trả về danh sách item được gợi ý
    return {
        "recommended_items": top_items["itemid"].tolist()
    }

async def demo_model_build():
    test_recommend = await recommend("05235448", 10)
    print(test_recommend)

@app.on_event("startup")
# Hàm khởi động thread xây dựng lại mô hình
async def start_background_model_rebuild():
    """
    Chạy background thread thực hiện xây dựng lại mô hình.
    """
    rebuild_interval = 300  # 5 minutes
    rebuild_thread = threading.Thread(
        target=periodic_model_rebuild,
        args=(rebuild_interval, train_and_save_model),
        daemon=True  # Ensures thread exits when main program exits
    )
    # rebuild_thread = threading.Thread(
    #     target=demo_model_build,
    #     args=(),
    #     daemon=True  # Ensures thread exits when main program exits
    # )
    rebuild_thread.start()