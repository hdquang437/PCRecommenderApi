import tensorflow as tf
import tensorflow_recommenders as tfrs
from tensorflow import keras
from keras import layers

vocab_sizes = {
    "type": 5,
    "location": 5,
    "gender": 2,
    "age_range": 10,
    "price_range": 5,
}

class WideAndDeepModel(tfrs.Model):
    def __init__(self):
        super().__init__()

        # Embedding layers cho categorical features
        self.type_embedding = layers.Embedding(input_dim=vocab_sizes["type"], output_dim=8)
        self.location_embedding = layers.Embedding(input_dim=vocab_sizes["location"], output_dim=8)
        self.gender_embedding = layers.Embedding(input_dim=vocab_sizes["gender"], output_dim=4)
        self.age_embedding = layers.Embedding(input_dim=vocab_sizes["age_range"], output_dim=4)
        self.price_embedding = layers.Embedding(input_dim=vocab_sizes["price_range"], output_dim=4)

        # Mô hình Wide (tuyến tính)
        self.wide = keras.Sequential([
            layers.Dense(1, activation='linear')
        ])

        # Mô hình Deep (MLP)
        self.deep = keras.Sequential([
            layers.Dense(128, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(1, activation="sigmoid")
        ])

        self.task = tfrs.tasks.Ranking(loss=tf.keras.losses.BinaryCrossentropy())

    def call(self, features):

        # Xử lý feature số (dùng cho cả Wide & Deep)
        numerical_features = tf.concat([
            tf.reshape(tf.cast(features["click_times"], tf.float32), (-1, 1)),
            tf.reshape(tf.cast(features["buy_times"], tf.float32), (-1, 1)),
            tf.reshape(tf.cast(features["rating"], tf.float32), (-1, 1))
        ], axis=1)

        # Lấy embeddings cho categorical features
        type_embedded = self.type_embedding(tf.cast(features["type"], tf.int32))
        location_embedded = self.location_embedding(tf.cast(features["location"], tf.int32))
        gender_embedded = self.gender_embedding(tf.cast(features["gender"], tf.int32))
        age_embedded = self.age_embedding(tf.cast(features["age_range"], tf.int32))
        price_embedded = self.price_embedding(tf.cast(features["price_range"], tf.int32))

        deep_input = tf.concat([
            tf.reshape(gender_embedded, (-1, 4)),  
            tf.reshape(age_embedded, (-1, 4)),
            tf.reshape(type_embedded, (-1, 8)),  
            tf.reshape(price_embedded, (-1, 4)), 
            tf.reshape(location_embedded, (-1, 8)),     
            numerical_features  # Dùng feature số cho cả Deep Model
        ], axis=1)

        # Wide Model (Dùng trực tiếp feature số)
        wide_output = self.wide(numerical_features)

        deep_output = self.deep(deep_input)
        output = tf.cast(wide_output + deep_output, tf.float32)

        return output
    
    def compute_loss(self, features, training=False):
        """Tính toán loss dựa trên đầu ra của model và label thực tế."""

        inputs, labels = features  # Giải nén tuple

        labels = tf.cast(labels, tf.float32)
        predictions = self.call(inputs)
        return self.task(labels=labels, predictions=predictions)
    
    def get_config(self):
        """Trả về config khi lưu model."""
        config = super(WideAndDeepModel, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        """Tạo lại model từ config nhưng bỏ các tham số không cần thiết."""
        config.pop("trainable", None)  # Loại bỏ tham số trainable
        return cls(**config)
