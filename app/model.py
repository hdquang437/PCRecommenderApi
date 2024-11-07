import tensorflow as tf
import pandas as pd

class WideAndDeepModel:
    def __init__(self):
        self.model = None
    
    def build_model(self, user_features, item_features):
        user_input = tf.keras.layers.Input(shape=(len(user_features),), name="user_features")
        item_input = tf.keras.layers.Input(shape=(len(item_features),), name="item_features")

        wide_part = tf.keras.layers.Dense(64, activation="relu")(user_input)
        deep_part = tf.keras.layers.Dense(64, activation="relu")(item_input)
        concat = tf.keras.layers.concatenate([wide_part, deep_part])

        output = tf.keras.layers.Dense(1, activation="sigmoid")(concat)
        self.model = tf.keras.Model(inputs=[user_input, item_input], outputs=output)
        self.model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    def train(self, X_train, y_train, epochs=10):
        if self.model:
            self.model.fit(X_train, y_train, epochs=epochs, batch_size=32)
    
    def predict(self, X):
        return self.model.predict(X) if self.model else None
