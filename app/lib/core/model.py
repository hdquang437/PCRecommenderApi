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

class AttentionLayer(layers.Layer):
    """
    Attention mechanism cho Deep component
    Giúp model tự động học được features nào quan trọng nhất
    """
    def __init__(self, attention_dim=8, name="attention_layer"):
        super(AttentionLayer, self).__init__(name=name)
        self.attention_dim = attention_dim
        
    def build(self, input_shape):
        """Build attention weights"""
        # Ma trận W: transform features to attention space
        self.W = self.add_weight(
            shape=(input_shape[-1], self.attention_dim),
            initializer='glorot_uniform',
            name='attention_W',
            trainable=True
        )
        
        # Bias b: điều chỉnh baseline
        self.b = self.add_weight(
            shape=(self.attention_dim,),
            initializer='zeros',
            name='attention_b',
            trainable=True
        )
        
        # Vector v: tính attention score final
        self.v = self.add_weight(
            shape=(self.attention_dim, 1),
            initializer='glorot_uniform',
            name='attention_v',
            trainable=True
        )
        
        super(AttentionLayer, self).build(input_shape)
        
    def call(self, inputs, return_attention=False):
        """
        Forward pass của attention mechanism
        
        Args:
            inputs: [batch_size, feature_dim] - Deep features
            return_attention: Bool - có trả về attention weights không
            
        Returns:
            attended_output: [batch_size, feature_dim] - Features sau khi apply attention
            attention_weights: [batch_size, feature_dim] - Attention weights (nếu return_attention=True)
        """
        # Bước 1: Transform features vào attention space
        attention_scores = tf.nn.tanh(tf.matmul(inputs, self.W) + self.b)
        # Shape: [batch_size, attention_dim]
        
        # Bước 2: Tính attention weights cho từng feature
        # Thay đổi để tạo attention weights cho từng feature dimension
        feature_dim = tf.shape(inputs)[1]
        
        # Tạo attention weights cho từng feature
        raw_weights = tf.matmul(attention_scores, self.v)  # [batch_size, 1]
        
        # Broadcast to feature dimension
        attention_weights = tf.nn.softmax(tf.broadcast_to(raw_weights, [tf.shape(inputs)[0], feature_dim]), axis=1)
        # Shape: [batch_size, feature_dim]
        
        # Bước 3: Apply attention weights lên original features
        attended_output = inputs * attention_weights
        # Shape: [batch_size, feature_dim]
        
        if return_attention:
            return attended_output, attention_weights
        
        return attended_output
    
    def get_config(self):
        """Config cho serialization"""
        config = super(AttentionLayer, self).get_config()
        config.update({
            'attention_dim': self.attention_dim
        })
        return config

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

        # ATTENTION MECHANISM for Deep component
        self.attention_layer = AttentionLayer(
            attention_dim=8, 
            name="feature_attention"
        )

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
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[
                tf.keras.metrics.RootMeanSquaredError(),
                tf.keras.metrics.MeanAbsoluteError(),
                tf.keras.metrics.R2Score(name="r2_score")
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
        
        wide_input = tf.concat([
            type_onehot, location_onehot, gender_onehot, age_onehot, price_onehot,
            gender_age_cross, type_price_cross,  # Cross features
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

        # Apply attention mechanism to deep features
        attended_input = self.attention_layer(deep_input)

        # Forward pass
        wide_output = self.wide(wide_input)
        deep_output = self.deep(attended_input, training=training)

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
    
    def get_attention_weights(self, features):
        """
        Lấy attention weights để phân tích patterns
        Useful để debug và hiểu model đang focus vào features nào
        """
        if isinstance(features, tuple):
            features, _ = features
            
        # Numerical features với normalization
        numerical_features = tf.concat([
            tf.reshape(tf.cast(features["click_times"], tf.float32), (-1, 1)),
            tf.reshape(tf.cast(features["rating"], tf.float32), (-1, 1))
        ], axis=1)
        
        # Simple normalization
        numerical_features = tf.clip_by_value(numerical_features, 0.0, 10.0) / 10.0
        
        # Safe index clipping
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

        # Get attention weights
        attended_output, attention_weights = self.attention_layer(deep_input, return_attention=True)
        
        return {
            'attention_weights': attention_weights,
            'attended_features': attended_output,
            'original_features': deep_input,
            'feature_names': ['gender(2)', 'age(2)', 'type(4)', 'price(2)', 'location(4)', 'numerical(2)']
        }
    
    def analyze_attention_patterns(self, sample_features, top_k=5):
        """
        Phân tích attention patterns để hiểu model behavior
        """
        attention_info = self.get_attention_weights(sample_features)
        attention_weights = attention_info['attention_weights'].numpy()
        
        # Tính average attention weights across all samples
        avg_attention = attention_weights.mean(axis=0)
        
        print("🔍 Attention Pattern Analysis:")
        print(f"Attention weights shape: {attention_weights.shape}")
        print(f"Average attention weights shape: {avg_attention.shape}")
        print(f"Average attention weights (first 6): {avg_attention[:6]}")
        
        # Feature importance ranking - Deep input structure:
        # [gender(2), age(2), type(4), price(2), location(4), numerical(2)]
        feature_segments = [
            ('gender', 0, 2),
            ('age', 2, 4), 
            ('type', 4, 8),
            ('price', 8, 10),
            ('location', 10, 14),
            ('numerical', 14, 16)
        ]
        
        feature_importance = []
        for name, start, end in feature_segments:
            # Tính importance cho từng feature segment
            if end <= len(avg_attention):
                # Tính tổng attention weights cho feature segment này
                importance = float(avg_attention[start:end].sum())
            else:
                importance = 0.0
            feature_importance.append((name, importance))
        
        # Sort by importance
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        print("\n📊 Feature Importance Ranking:")
        for i, (name, importance) in enumerate(feature_importance[:top_k]):
            print(f"  {i+1}. {name}: {importance:.4f}")
        
        # Thêm thông tin chi tiết về attention distribution
        total_attention = sum([imp for _, imp in feature_importance])
        print(f"\n📈 Attention Distribution:")
        for name, importance in feature_importance:
            percentage = (importance / total_attention * 100) if total_attention > 0 else 0
            print(f"  {name}: {importance:.4f} ({percentage:.1f}%)")
        
        return feature_importance
