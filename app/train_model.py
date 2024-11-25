import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from .data_processing import update_main_data, load_main_data
from .paths import MODEL_PATH

# Hàm load dữ liệu từ CSV
def load_data():
    # cập nhật lại dữ liệu từ cache
    update_main_data()
    # load dữ liệu
    user_data, item_data, interaction_data = load_main_data()
    return user_data, item_data, interaction_data

# Chuẩn hóa dữ liệu
def normalize_features(interaction_data):
    scaler = MinMaxScaler()
    interaction_data[["click_times", "purchase_times", "rating", "is_favorite"]] = scaler.fit_transform(
        interaction_data[["click_times", "purchase_times", "rating", "is_favorite"]]
    )
    return interaction_data

# Hàm xây dựng Wide & Deep Model
def build_wide_and_deep_model(user_data, item_data):
    # Input layer
    user_id_input = tf.keras.Input(shape=(1,), name="user_id")
    item_id_input = tf.keras.Input(shape=(1,), name="item_id")
    interaction_features_input = tf.keras.Input(shape=(4,), name="interaction_features")
    user_features_input = tf.keras.Input(shape=(user_data.shape[1] - 1,), name="user_features")  # Trừ cột 'userid'
    item_features_input = tf.keras.Input(shape=(item_data.shape[1] - 1,), name="item_features")  # Trừ cột 'itemid'

    # Embedding layers
    user_embedding = tf.keras.layers.Embedding(input_dim=1000, output_dim=32)(user_id_input)
    item_embedding = tf.keras.layers.Embedding(input_dim=1000, output_dim=32)(item_id_input)

    # Wide part
    wide_input = tf.keras.layers.Concatenate()([user_features_input, item_features_input, interaction_features_input])

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
    user_ids = interaction_data["userid"].values
    item_ids = interaction_data["itemid"].values
    interaction_features = normalize_features(interaction_data)

    # Kết hợp đặc trưng user và item
    user_features = user_data.set_index("userid").reindex(user_ids).fillna(0).values
    item_features = item_data.set_index("itemid").reindex(item_ids).fillna(0).values

    # Chuẩn bị đầu vào cho mô hình
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
    model = build_wide_and_deep_model(user_data, item_data)

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

# Thực thi train
if __name__ == "__main__":
    train_and_save_model()
