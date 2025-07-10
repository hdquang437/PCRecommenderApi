import os
import warnings

# Tắt TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import tensorflow as tf
import tensorflow_recommenders as tfrs
from tensorflow import keras
from keras import layers

# Cấu hình TensorFlow logging
tf.get_logger().setLevel('FATAL')
tf.autograph.set_verbosity(0)

def focal_loss(gamma=2.0, alpha=0.25):
    def loss_fn(y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_factor = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
        focal_weight = alpha_factor * tf.pow(1. - p_t, gamma)
        loss = -focal_weight * tf.math.log(p_t)

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, tf.float32)
            loss *= sample_weight

        return tf.reduce_mean(loss)

    return loss_fn

class WideAndDeepModel(tfrs.Model):
    def __init__(self, vocab_sizes=None, name="wide_and_deep_model", *args, **kwargs):
        super(WideAndDeepModel, self).__init__(name=name, *args, **kwargs)
        
        if vocab_sizes is None:
            vocab_sizes = {
                "type": 30, "location": 63, "gender": 2,
                "age_range": 5, "price_range": 5,
            }
        
        self.vocab_sizes = vocab_sizes

        # Embeddings cho Deep part - REDUCED DIMENSIONS
        self.type_embedding = layers.Embedding(
            input_dim=vocab_sizes["type"], output_dim=4, name="type_embedding"
        )
        self.location_embedding = layers.Embedding(
            input_dim=vocab_sizes["location"], output_dim=4, name="location_embedding"
        )
        self.gender_embedding = layers.Embedding(
            input_dim=vocab_sizes["gender"], output_dim=2, name="gender_embedding"
        )
        self.age_embedding = layers.Embedding(
            input_dim=vocab_sizes["age_range"], output_dim=2, name="age_embedding"
        )
        self.price_embedding = layers.Embedding(
            input_dim=vocab_sizes["price_range"], output_dim=2, name="price_embedding"
        )

        # Wide component - Linear layer
        self.wide = layers.Dense(1, activation='linear', use_bias=True, name="wide_layer")
        
        # Combination layer để kết hợp wide và deep outputs với sigmoid activation
        self.combination_layer = layers.Dense(1, activation='sigmoid', name="combination_layer")

        # Deep component - Balanced regularization
        self.deep = keras.Sequential([
            layers.Dense(32, activation="relu", name="deep_dense_1",
                        kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
            layers.Dropout(0.2, name="deep_dropout_1"),
            layers.Dense(16, activation="relu", name="deep_dense_2",
                        kernel_regularizer=tf.keras.regularizers.l2(0.0005)), 
            layers.Dropout(0.1, name="deep_dropout_2"),
            layers.Dense(1, activation='linear', name="deep_output")
        ], name="deep_network")

        self.task = tfrs.tasks.Ranking(
            loss=focal_loss(gamma=2.0, alpha=0.25),
            metrics=[
                tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
                tf.keras.metrics.AUC(name="auc"),
                tf.keras.metrics.F1Score(name="f1_score")
            ],
            name="ranking_task"
        )

        self.wide_weight = self.add_weight(
            name="wide_weight", shape=(), initializer=tf.constant_initializer(0.5), trainable=True
        )
        self.deep_weight = self.add_weight(
            name="deep_weight", shape=(), initializer=tf.constant_initializer(0.5), trainable=True
        )

    def build(self, input_shape):
        """Build the model layers."""
        super(WideAndDeepModel, self).build(input_shape)
        # Các layers đã được khởi tạo trong __init__, 
        # method này chỉ để đánh dấu model đã được built
        self.built = True

    def call(self, features, training=None):
        if isinstance(features, tuple):
            features, _ = features
            
        # Numerical features với normalization
        numerical_features = tf.concat([
            tf.reshape(tf.cast(features["click_times"], tf.float32), (-1, 1)),
            tf.reshape(tf.cast(features["rating"], tf.float32), (-1, 1))
        ], axis=1)
        
        # Simple normalization để tránh extreme values
        numerical_features = tf.clip_by_value(numerical_features, 0.0, 10.0) / 10.0        # One-hot cho Wide part với SAFE INDEX CLIPPING
        type_indices = tf.reshape(tf.cast(features["type"], tf.int32), [-1])
        type_indices = tf.clip_by_value(type_indices, 0, self.vocab_sizes["type"] - 1)
        
        location_indices = tf.reshape(tf.cast(features["location"], tf.int32), [-1])
        location_indices = tf.clip_by_value(location_indices, 0, self.vocab_sizes["location"] - 1)
        
        gender_indices = tf.reshape(tf.cast(features["gender"], tf.int32), [-1])
        gender_indices = tf.clip_by_value(gender_indices, 0, self.vocab_sizes["gender"] - 1)
        
        age_indices = tf.reshape(tf.cast(features["age_range"], tf.int32), [-1])
        age_indices = tf.clip_by_value(age_indices, 0, self.vocab_sizes["age_range"] - 1)
        
        price_indices = tf.reshape(tf.cast(features["price_range"], tf.int32), [-1])
        price_indices = tf.clip_by_value(price_indices, 0, self.vocab_sizes["price_range"] - 1)

        type_onehot = tf.one_hot(type_indices, self.vocab_sizes["type"])
        location_onehot = tf.one_hot(location_indices, self.vocab_sizes["location"])
        gender_onehot = tf.one_hot(gender_indices, self.vocab_sizes["gender"])
        age_onehot = tf.one_hot(age_indices, self.vocab_sizes["age_range"])
        price_onehot = tf.one_hot(price_indices, self.vocab_sizes["price_range"])

        # Wide input: với cross products cho memorization  
        # Tạo cross features cho gender x age_range và type x price_range
        gender_age_cross = tf.reshape(
            tf.einsum('bi,bj->bij', gender_onehot, age_onehot),
            [-1, self.vocab_sizes["gender"] * self.vocab_sizes["age_range"]]
        )
        
        type_price_cross = tf.reshape(
            tf.einsum('bi,bj->bij', type_onehot, price_onehot),
            [-1, self.vocab_sizes["type"] * self.vocab_sizes["price_range"]]
        )

        # Age × Price (Độ tuổi × Giá)
        age_price_cross = tf.reshape(
            tf.einsum('bi,bj->bij', age_onehot, price_onehot),
            [-1, self.vocab_sizes["age_range"] * self.vocab_sizes["price_range"]]
        )

        # Type × Gender (Loại sản phẩm × Giới tính)
        type_gender_cross = tf.reshape(
            tf.einsum('bi,bj->bij', type_onehot, gender_onehot),
            [-1, self.vocab_sizes["type"] * self.vocab_sizes["gender"]]
        )
        
        wide_input = tf.concat([
            type_onehot, location_onehot, gender_onehot, age_onehot, price_onehot,
            gender_age_cross, type_price_cross, age_price_cross, type_gender_cross,  # Cross features
            numerical_features
        ], axis=1)

        # Embeddings cho Deep part
        type_embedded = self.type_embedding(type_indices)
        location_embedded = self.location_embedding(location_indices)
        gender_embedded = self.gender_embedding(gender_indices)
        age_embedded = self.age_embedding(age_indices)
        price_embedded = self.price_embedding(price_indices)

        # Deep input
        deep_input = tf.concat([
            tf.reshape(gender_embedded, [-1, 2]),
            tf.reshape(age_embedded, [-1, 2]),
            tf.reshape(type_embedded, [-1, 4]),
            tf.reshape(price_embedded, [-1, 2]),
            tf.reshape(location_embedded, [-1, 4]),
            numerical_features
        ], axis=1)

        # Forward pass
        wide_output = self.wide(wide_input)
        deep_output = self.deep(deep_input, training=training)

        # Scale và cộng weighted sum rồi sigmoid
        combined = self.wide_weight * wide_output + self.deep_weight * deep_output
        combined_output = tf.keras.activations.sigmoid(combined)

        return combined_output

    def compute_loss(self, features, training=False):
        inputs, labels = features
        labels = tf.cast(labels, tf.float32)
        labels = tf.reshape(labels, [-1, 1])
        predictions = self.call(inputs, training=training)
        
        return self.task(labels=labels, predictions=predictions)
