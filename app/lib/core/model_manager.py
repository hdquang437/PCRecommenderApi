import tensorflow as tf
import os
from .model import WideAndDeepModel
from ...paths import MODEL_PATH

class ModelManager:
    _instance = None  # Biến lưu instance duy nhất của class

    def __new__(cls):
        """Singleton: Đảm bảo chỉ có một instance duy nhất."""
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance.model = None
        return cls._instance

    def load_model(self, vocab_sizes=None, reload=False):
        """Load mô hình từ file hoặc tạo mới nếu chưa có."""

        if self.model is None or reload:
            if os.path.exists(MODEL_PATH):
                print("Loading existing model...")
                self.model = tf.keras.models.load_model(MODEL_PATH, custom_objects={"WideAndDeepModel": WideAndDeepModel})
            else:
                print("Creating new model...")
                self.model = WideAndDeepModel(vocab_sizes=vocab_sizes)
                self.model.compile(optimizer=tf.keras.optimizers.Adam())

        return self.model

    def save_model(self):
        """Lưu mô hình vào file."""
        if self.model is not None:
            self.model.save(MODEL_PATH)
            print("Model saved!")

    def get_model(self):
        """Trả về mô hình hiện tại."""
        if self.model is None:
            raise ValueError("Model chưa được load. Hãy gọi load_model() trước.")
        return self.model
