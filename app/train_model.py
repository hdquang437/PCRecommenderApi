import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from .data_processing import update_main_data, load_main_data, cache_fit_ids
from .paths import MODEL_PATH
import time

# Hàm load dữ liệu từ CSV
def load_data():
    # cập nhật lại dữ liệu từ cache
    update_main_data()
    # load dữ liệu
    user_data, item_data, interaction_data = load_main_data()
    return user_data, item_data, interaction_data

# Chuẩn hóa dữ liệu
def encode_item_features(item_data):
    item_type_column = item_data["item_type"].values
    # Tạo một LabelEncoder
    label_encoder = LabelEncoder()
    # Chuyển đổi cột chuỗi thành các giá trị số
    item_type_column_encoded = label_encoder.fit_transform(item_type_column)
    # Gán lại vào mảng NumPy
    item_data["item_type"] = item_type_column_encoded
    return item_data

def normalize_features(interaction_data):
    scaler = MinMaxScaler()
    interaction_data[["click_times", "purchase_times", "rating", "is_favorite"]] = scaler.fit_transform(
        interaction_data[["click_times", "purchase_times", "rating", "is_favorite"]]
    )
    return interaction_data

# Hàm xây dựng Wide & Deep Model
def build_wide_and_deep_model(user_data, item_data, interaction_data):
    # Input layer
    user_id_input = tf.keras.Input(shape=(1,), name="user_id")
    item_id_input = tf.keras.Input(shape=(1,), name="item_id")
    interaction_features_input = tf.keras.Input(shape=(interaction_data.shape[1] - 2,), name="interaction_features") # Trừ cột "userid" và "itemid"
    user_features_input = tf.keras.Input(shape=(user_data.shape[1] - 1,), name="user_features")  # Trừ cột 'userid'
    item_features_input = tf.keras.Input(shape=(item_data.shape[1] - 1,), name="item_features")  # Trừ cột 'itemid'

    user_max_index = len(user_data["userid"].unique())
    print(f"Max index in user_data: {user_max_index}")
    item_max_index = len(item_data["itemid"].unique())
    print(f"Max index in item_data: {item_max_index}")
    # Embedding layers
    user_embedding = tf.keras.layers.Embedding(input_dim=user_max_index, output_dim=32)(user_id_input)
    item_embedding = tf.keras.layers.Embedding(input_dim=item_max_index, output_dim=32)(item_id_input)

    # Wide part
    wide_input = tf.keras.layers.Concatenate()([user_features_input, item_features_input])

    # Deep part
    deep_input = tf.keras.layers.Concatenate()([tf.keras.layers.Flatten()(user_embedding), tf.keras.layers.Flatten()(item_embedding), interaction_features_input])
    deep_dense = tf.keras.layers.Dense(128, activation="relu")(deep_input)
    deep_output = tf.keras.layers.Dense(64, activation="relu")(deep_dense)

    # Combine wide and deep
    combined = tf.keras.layers.Concatenate()([wide_input, deep_output])
    output = tf.keras.layers.Dense(1)(combined)

    # Model
    model = tf.keras.Model(inputs=[user_id_input, item_id_input, interaction_features_input, user_features_input, item_features_input], outputs=output)
    return model

# Hàm train và lưu mô hình
def train_and_save_model():
    user_data, item_data, interaction_data = load_data()

    # Chuẩn bị dữ liệu tương tác
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()

    user_ids = user_encoder.fit_transform(interaction_data["userid"].values)
    item_ids = item_encoder.fit_transform(interaction_data["itemid"].values)
    interaction_features = normalize_features(interaction_data)
    item_data = encode_item_features(item_data)

    # Kết hợp đặc trưng user và item
    user_features = user_data.reindex(user_ids).fillna(0).values
    item_features = item_data.reindex(item_ids).fillna(0).values
    cache_fit_ids(user_features, item_features)

    # Chuẩn bị đầu vào cho mô hình
    user_ids = user_ids.astype(np.int64)
    item_ids = item_ids.astype(np.int64)
    interaction_features = interaction_features.iloc[:, 2:interaction_features.shape[1]].astype(np.int64)
    user_features = user_features[:, 1:user_features.shape[1]].astype(np.int64)
    item_features = item_features[:, 1:item_features.shape[1]].astype(np.int64)

    input_data = {
        "user_id": user_ids,
        "item_id": item_ids,
        "interaction_features": interaction_features,
        "user_features": user_features,
        "item_features": item_features,
    }

    # Chuẩn bị nhãn
    labels = interaction_data["rating"].values  # Hoặc một cột khác thể hiện mục tiêu

    # Xây dựng mô hình Wide & Deep
    model = build_wide_and_deep_model(user_data, item_data, interaction_data)

    # Biên dịch mô hình
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    # Huấn luyện mô hình
    model.fit(
        [input_data["user_id"], input_data["item_id"], input_data["interaction_features"], input_data["user_features"], input_data["item_features"]],
        labels,
        epochs=10,
        batch_size=32,
        validation_split=0.2,
    )

    # Lưu mô hình
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

# Hàm build lại model theo chu kỳ
def periodic_model_rebuild(interval, model_rebuild_function):
    """
    Xây dựng lại mô hình theo chu kỳ.

    Args:
        interval (int): Chu kỳ tính theo giây để chờ xây dựng lại mô hình.
        model_rebuild_function (function): Hàm build lại mô hình
    """
    while True:
        # Ghi lại thời gian bắt đầu
        start_time = time.time()

        print("Rebuilding model...")
        model_rebuild_function()
        
        # Ghi lại thời gian kết thúc
        end_time = time.time()
        
        # Tính toán thời gian thực thi
        execution_time = end_time - start_time

        print(f"Model rebuild completed [{execution_time:.4f} seconds]. Waiting for next cycle...")
        
        time.sleep(interval)

        

# Thực thi train
if __name__ == "__main__":
    train_and_save_model()
