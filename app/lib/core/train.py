import os
import shutil
import warnings

# Tắt TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow import keras
from keras import layers

# Cấu hình TensorFlow logging
tf.get_logger().setLevel('FATAL')
tf.autograph.set_verbosity(0)

from .data_manager import DataManager
from .model_manager import ModelManager

def train_model():
    BATCH_SIZE = 8
    TRAIN_RATIO = 0.7

    try:
        data_manager = DataManager() 
        data = data_manager.get_data()
        dataset = data_manager.get_dataset()

        print(f"Total data size: {len(data)}")
        
        if len(data) < 20:
            return {
                "status": "error",
                "error": "Dataset too small (< 20 samples). Need more data to train."
            }

        train_size = int(TRAIN_RATIO * len(data))
        test_size = len(data) - train_size
        
        print(f"Train size: {train_size}, Test size: {test_size}")

        # FIX: Split first, then batch, then repeat for small datasets
        train_ds = dataset.take(train_size)
        test_ds = dataset.skip(train_size)
        
        # Calculate steps properly - be more conservative
        train_steps = max(1, min(train_size // BATCH_SIZE, train_size))  # Don't exceed actual data
        val_steps = max(1, min(test_size // BATCH_SIZE, test_size))
        
        print(f"Train steps: {train_steps}, Validation steps: {val_steps}")
        print(f"Batch size: {BATCH_SIZE}")
        
        # Batch and repeat for small datasets
        train_ds = train_ds.batch(BATCH_SIZE).repeat()
        test_ds = test_ds.batch(BATCH_SIZE).repeat()

        # ===== FORCE CLEAN REBUILD =====
        model_manager = ModelManager()
        vocab_sizes = data_manager.get_vocab_sizes()
        
        # Delete old model directory completely
        model_dir = os.path.join(os.path.dirname(__file__), '..', 'wide_deep_model')
        if os.path.exists(model_dir):
            try:
                shutil.rmtree(model_dir)
                print(f"Removed old model directory: {model_dir}")
            except Exception as e:
                print(f"Warning: Could not remove model dir: {e}")
        
        # Force create new model (bypass loading)
        print("Creating new model from scratch...")
        model_manager.model = None  # Reset any cached model
        
        # Create brand new model
        model = model_manager.load_model(vocab_sizes=vocab_sizes, reload=True, force_new=True)        # Compile with new settings
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            run_eagerly=False
        )

        # Train model with proper steps
        history = model.fit(
            train_ds, 
            epochs=3,
            steps_per_epoch=train_steps,
            validation_data=test_ds,
            validation_steps=val_steps,
            verbose=1
        )        # Test predictions - create a fresh dataset for testing
        sample_test_ds = dataset.take(5).batch(5)  # Take 5 samples for testing
        sample_batch = next(iter(sample_test_ds))
        features, labels = sample_batch
        predictions = model(features)
        
        print(f"Sample predictions range: {predictions.numpy().min():.4f} - {predictions.numpy().max():.4f}")
        print(f"Sample predictions mean: {predictions.numpy().mean():.4f}")
        print(f"Sample predictions: {predictions.numpy().flatten()[:3]}")  # Show first 3        # Save new model
        model_manager.save_model()
        
        return {
            "status": "success",
            "train_size": train_size,
            "test_size": test_size,
            "train_steps": train_steps,
            "val_steps": val_steps,
            "final_loss": float(history.history['loss'][-1]),
            "final_val_loss": float(history.history.get('val_loss', [0])[-1]) if 'val_loss' in history.history else None,
            "note": "Model trained successfully with proper dataset configuration"
        }

    except Exception as e:
        print(f"Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": str(e)}
