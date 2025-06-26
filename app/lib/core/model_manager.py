import os
import warnings

# Tắt TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import tensorflow as tf

# Cấu hình TensorFlow logging
tf.get_logger().setLevel('FATAL')
tf.autograph.set_verbosity(0)

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

    def load_model(self, vocab_sizes=None, reload=False, force_new=False):
        """Load mô hình từ file hoặc tạo mới nếu chưa có."""

        if force_new:
            # Force create new model, ignore saved version
            print("Force creating new model...")
            self.model = WideAndDeepModel(vocab_sizes=vocab_sizes)
            return self.model
            
        if self.model is None or reload:
            if os.path.exists(MODEL_PATH) and not force_new:
                try:
                    print("Loading existing model...")
                    self.model = tf.keras.models.load_model(
                        MODEL_PATH, 
                        custom_objects={"WideAndDeepModel": WideAndDeepModel}
                    )
                    print("Model loaded successfully!")
                except Exception as e:
                    print(f"Failed to load existing model: {e}")
                    print("Creating new model...")
                    self.model = WideAndDeepModel(vocab_sizes=vocab_sizes)
            else:
                print("Creating new model...")
                self.model = WideAndDeepModel(vocab_sizes=vocab_sizes)
        
        return self.model

    def save_model(self):
        """Lưu mô hình vào file."""
        if self.model is not None:
            try:
                # Create directory if not exists
                os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
                
                # Save model
                self.model.save(MODEL_PATH)
                print("Model saved!")
            except Exception as e:
                print(f"Failed to save model: {e}")
        else:
            print("No model to save")
    
    def get_model(self):
        """Trả về mô hình hiện tại."""
        return self.model
