import tensorflow as tf
import pandas as pd
import numpy as np
from .data_manager import DataManager
from .model_manager import ModelManager

# Khởi tạo ModelManager và DataManager
model_manager = ModelManager()
data_manager = DataManager()

# Load model sau khi data manager đã được preprocessing
model = None

def _ensure_model_loaded():
    """Đảm bảo model được load với vocab_sizes đúng"""
    global model
    if model is None:
        # Đảm bảo data đã được load và preprocess
        if data_manager.data is None:
            raise ValueError("Data must be loaded before loading model")
        
        vocab_sizes = data_manager.get_vocab_sizes()
        model = model_manager.load_model(vocab_sizes=vocab_sizes)
    return model


def find_similar_products(target_product_id):
    """Tìm sản phẩm tương tự dựa trên type, price_range, location."""

    data = data_manager.get_data()

    target_product = data[data["product_id"] == target_product_id]

    if target_product.empty:
        print(f"Không tìm thấy sản phẩm {target_product_id}")
        return pd.DataFrame()

    target_type = target_product["type"].values[0]
    target_price = target_product["price_range"].values[0]
    target_location = target_product["location"].values[0]

    # Tìm sản phẩm cùng loại
    similar_products = data[data["type"] == target_type]

    # Ưu tiên cùng price_range và location
    similar_products["score"] = (
        (similar_products["price_range"] == target_price).astype(int) * 0.5 +
        (similar_products["location"] == target_location).astype(int) * 0.5
    )

    # Tổng hợp dữ liệu dựa trên toàn bộ user
    ranking = similar_products.groupby("product_id").agg(
        total_rating=("rating", "sum"),
        total_clicks=("click_times", "sum"),
        total_buys=("buy_times", "sum"),
        match_score=("score", "mean")  # Điểm tương đồng (cùng giá, cùng location)
    ).reset_index()

    # Xếp hạng sản phẩm dựa trên trọng số
    ranking["final_score"] = (
        ranking["total_rating"] * 0.3 +
        ranking["total_clicks"] * 0.3 +
        ranking["total_buys"] * 0.4 +
        ranking["match_score"] * 0.2  # Thêm điểm tương đồng vào tính toán
    )

    # Sắp xếp theo final_score
    ranking = ranking.sort_values(by="final_score", ascending=False)

    return ranking


def get_user_behavior_for_similar_products(user_id, target_product_id):
    """Lấy thông tin trung bình từ các sản phẩm tương tự mà user đã tương tác."""

    data = data_manager.get_data()

    target_product = data[data["product_id"] == target_product_id]

    if target_product.empty:
        print(f"Không tìm thấy sản phẩm {target_product_id}")
        return None

    target_type = target_product["type"].values[0]
    target_price = target_product["price_range"].values[0]
    target_location = target_product["location"].values[0]

    # Lọc các sản phẩm tương tự mà user đã từng tương tác
    similar_products = data[
        (data["user_id"] == user_id) &
        (data["type"] == target_type)
    ]

    # Ưu tiên sản phẩm có price_range và location giống nhau
    similar_products["match_score"] = (
        (similar_products["price_range"] == target_price).astype(int) * 0.5 +
        (similar_products["location"] == target_location).astype(int) * 0.5
    )

    # Nếu không có sản phẩm tương tự nào user đã tương tác, return None
    if similar_products.empty:
        return None

    # Tính trung bình rating, click_times, buy_times của user với nhóm sản phẩm tương tự
    behavior_avg = similar_products.agg({
        "rating": "mean",
        "click_times": "mean",
        "buy_times": "mean"
    }).to_dict()

    return behavior_avg


def predict(user_id, product_id):
    """Dự đoán khả năng user mua sản phẩm, ngay cả khi chưa từng tương tác với nó."""
    try:
        model = _ensure_model_loaded()  # Đảm bảo model được load
        data = data_manager.get_data()
        
        if data is None or data.empty:
            raise ValueError("No data available for prediction")
        
        sample = data[(data["user_id"] == user_id) & (data["product_id"] == product_id)]

        if not sample.empty:
            # Dữ liệu có sẵn, thực hiện dự đoán
            sample_dict = dict(sample.iloc[0])
            sample_dict.pop("label", None)  # Loại bỏ nhãn
            sample_dict.pop("user_id", None)  # Không cần user_id khi đưa vào model
            sample_dict.pop("product_id", None)  # Không cần product_id khi đưa vào model
            
            sample_ds = tf.data.Dataset.from_tensors(sample_dict).batch(1)
            
            prediction = model.predict(sample_ds)
            print(f"Xác suất user {user_id} mua {product_id}: {prediction[0][0]:.2%}")
            return prediction[0][0]

        print(f"User {user_id} chưa từng tương tác với sản phẩm {product_id}")

        # Tạo sample cho user chưa tương tác
        sample_dict = data_manager.build_empty_sample(user_id, product_id)
        
        # Mã hóa categorical fields trước khi tạo dataset
        sample_encoded = data_manager.encode_sample(sample_dict)
        
        # Loại bỏ các trường không cần thiết cho model
        sample_for_model = sample_encoded.copy()
        sample_for_model.pop("label", None)
        sample_for_model.pop("user_id", None)
        sample_for_model.pop("product_id", None)
        
        sample_ds = tf.data.Dataset.from_tensors(sample_for_model).batch(1)
        prediction = model.predict(sample_ds)
        print(f"Xác suất user {user_id} mua {product_id}: {prediction[0][0]:.2%}")
        return prediction[0][0]
        
    except Exception as e:
        print(f"Error in predict function for user {user_id}, product {product_id}: {e}")
        return -1.0  # Return negative value to indicate error

    # # Lấy thông tin từ sản phẩm tương tự user đã từng xem/mua
    # user_behavior = get_user_behavior_for_similar_products(user_id, product_id)

    # if user_behavior is None:
    #     print("User chưa từng tương tác với sản phẩm nào cùng loại. Không thể dự đoán.")
    #     return -1

    # # Tạo mẫu dữ liệu giả lập để đưa vào mô hình
    # target_product = data[data["product_id"] == product_id].iloc[0].to_dict()

    # # Gán thông tin user behavior vào
    # target_product.update(user_behavior)
    # target_product.pop("label")  # Loại bỏ nhãn
    # target_product.pop("user_id")  # Không cần user_id khi đưa vào model
    # target_product.pop("product_id")  # Không cần product_id khi đưa vào model

    # sample_ds = tf.data.Dataset.from_tensors(target_product).batch(1)

    # prediction = model.predict(sample_ds)
    # print(f"Xác suất user {user_id} mua {product_id}: {prediction[0][0]:.2%} (dựa trên sản phẩm tương tự)")
    # return prediction[0][0]

def get_top_popular_products(n=10):
    """Tìm n sản phẩm phổ biến nhất dựa trên dữ liệu thực tế."""
    data = data_manager.get_data()
    
    popular_products = data.groupby("product_id").agg(
        total_rating=("rating", "sum"),
        total_clicks=("click_times", "sum"),
        total_buys=("buy_times", "sum")
    ).reset_index()

    # Tính điểm tổng hợp (có thể điều chỉnh trọng số)
    popular_products["popularity_score"] = (
        popular_products["total_rating"] * 0.5 +
        popular_products["total_clicks"] * 0.1 +
        popular_products["total_buys"] * 0.4
    )

    # Sắp xếp theo độ phổ biến
    top_products = popular_products.sort_values(by="popularity_score", ascending=False)["product_id"].head(n).tolist()
    
    return top_products