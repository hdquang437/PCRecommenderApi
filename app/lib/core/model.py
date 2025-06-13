import tensorflow as tf
import tensorflow_recommenders as tfrs
from tensorflow import keras
from keras import layers

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

        # Deep component - MUCH SMALLER với heavy regularization
        self.deep = keras.Sequential([
            layers.Dense(16, activation="relu", name="deep_dense_1",
                        kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            layers.Dropout(0.8, name="deep_dropout_1"),
            layers.Dense(8, activation="relu", name="deep_dense_2",
                        kernel_regularizer=tf.keras.regularizers.l2(0.01)), 
            layers.Dropout(0.8, name="deep_dropout_2"),
            layers.Dense(1, activation='linear', name="deep_output")
        ], name="deep_network")

        self.task = tfrs.tasks.Ranking(
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
            name="ranking_task"
        )

    def call(self, features, training=None):
        if isinstance(features, tuple):
            features, _ = features
            
        # Numerical features với normalization
        numerical_features = tf.concat([
            tf.reshape(tf.cast(features["click_times"], tf.float32), (-1, 1)),
            tf.reshape(tf.cast(features["buy_times"], tf.float32), (-1, 1)),
            tf.reshape(tf.cast(features["rating"], tf.float32), (-1, 1))
        ], axis=1)
        
        # Simple normalization để tránh extreme values
        numerical_features = tf.clip_by_value(numerical_features, 0.0, 10.0) / 10.0

        # One-hot cho Wide part
        type_indices = tf.reshape(tf.cast(features["type"], tf.int32), [-1])
        location_indices = tf.reshape(tf.cast(features["location"], tf.int32), [-1])
        gender_indices = tf.reshape(tf.cast(features["gender"], tf.int32), [-1])
        age_indices = tf.reshape(tf.cast(features["age_range"], tf.int32), [-1])
        price_indices = tf.reshape(tf.cast(features["price_range"], tf.int32), [-1])

        type_onehot = tf.one_hot(type_indices, self.vocab_sizes["type"])
        location_onehot = tf.one_hot(location_indices, self.vocab_sizes["location"])
        gender_onehot = tf.one_hot(gender_indices, self.vocab_sizes["gender"])
        age_onehot = tf.one_hot(age_indices, self.vocab_sizes["age_range"])
        price_onehot = tf.one_hot(price_indices, self.vocab_sizes["price_range"])

        # Wide input: SIMPLE concatenation, no cross products để tránh overfitting
        wide_input = tf.concat([
            type_onehot, location_onehot, gender_onehot, age_onehot, price_onehot,
            numerical_features
        ], axis=1)

        # Embeddings cho Deep part - SMALLER
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

        # MORE AGGRESSIVE combination
        combined_output = 0.2 * wide_output + 0.8 * deep_output
        
        # Much more aggressive temperature
        output = tf.nn.sigmoid(combined_output / 0.5)  # Temperature = 0.5 (very aggressive)

        return output
    
    def compute_loss(self, features, training=False):
        inputs, labels = features
        labels = tf.cast(labels, tf.float32)
        labels = tf.reshape(labels, [-1, 1])
        predictions = self.call(inputs, training=training)
        
        # Reduce label smoothing để có more confident predictions
        smoothed_labels = labels * 0.98 + 0.01  # Less smoothing
        
        return self.task(labels=smoothed_labels, predictions=predictions)
