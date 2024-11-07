# app/train_model.py
from app.data_processing import load_data, encode_data, prepare_data
from app.model import build_wide_and_deep_model
import tensorflow as tf

def train_model():
    user_data, item_data, interaction_data = load_data()
    user_data, item_data, user_encoder, item_encoder = encode_data(user_data, item_data)
    data = prepare_data(user_data, item_data, interaction_data)

    # Xây dựng mô hình
    model = build_wide_and_deep_model()

    # Chia dữ liệu thành train và test
    train_data = data.sample(frac=0.8, random_state=42)
    test_data = data.drop(train_data.index)

    train_labels = train_data['rating']
    test_labels = test_data['rating']

    # Huấn luyện mô hình
    model.fit(train_data, train_labels, epochs=5, batch_size=32, validation_data=(test_data, test_labels))

    # Lưu mô hình đã huấn luyện
    model.save('wide_deep_model.h5')

if __name__ == "__main__":
    train_model()