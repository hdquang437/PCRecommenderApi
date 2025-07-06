import os
import shutil
import warnings
import random

# Tắt TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np

# Cấu hình TensorFlow logging
tf.get_logger().setLevel('FATAL')
tf.autograph.set_verbosity(0)

from .data_manager import DataManager
from .model_manager import ModelManager
from ...paths import MODEL_PATH
from .model_config import BATCH_SIZE, RANDOM_SEED, USE_CSV

def train_model():
    TRAIN_RATIO = 0.7

    # Global random seeds
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    try:
        data_manager = DataManager() 
        data = data_manager.get_data()
        dataset = data_manager.get_dataset()
        dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

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

        # ===== SMART MODEL LOADING =====
        model_manager = ModelManager()
        vocab_sizes = data_manager.get_vocab_sizes()
        
        # ===== INCREMENTAL TRAINING RE-ENABLED FOR TESTING =====
        # TODO_INCREMENTAL: Testing incremental training again
        # Previous issues: Unstable correlation results (oscillating between +0.61 and -0.41)
        # Now testing if data consistency improvements help
        
        # INCREMENTAL TRAINING LOGIC (RE-ENABLED):
        model_file = MODEL_PATH
        if os.path.exists(model_file):
            try:
                print("Attempting to load existing SAVED model for incremental training...")
                model = model_manager.load_model(vocab_sizes=vocab_sizes, reload=True)
                print("✅ Loaded existing SAVED model - will continue training")
                incremental_training = True
            except Exception as e:
                print(f"Failed to load saved model: {e}")
                print("Creating new model from scratch...")
                
                # Clear any existing models
                model_dir = os.path.dirname(MODEL_PATH)
                if os.path.exists(model_dir):
                    try:
                        shutil.rmtree(model_dir)
                        print(f"Removed old model directory: {model_dir}")
                    except Exception as e:
                        print(f"Warning: Could not remove model dir: {e}")
                
                model_manager.model = None
                model = model_manager.load_model(vocab_sizes=vocab_sizes, reload=True, force_new=True)
                incremental_training = False
        else:
            print("No saved model found - creating new model from scratch...")
            model_manager.model = None
            model = model_manager.load_model(vocab_sizes=vocab_sizes, reload=True, force_new=True)
            incremental_training = False
        
        learning_rate = 0.0002 if incremental_training else 0.0005
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            run_eagerly=False
        )
        
        print(f"Training mode: {'Incremental' if incremental_training else 'From scratch'}")
        print(f"Learning rate: {learning_rate}")

        # Train model with early stopping for better continuous learning
        # Use adaptive training parameters based on mode
        max_epochs = 3 if incremental_training else 7
        patience = 1 if incremental_training else 2
        
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )

        history = model.fit(
            train_ds, 
            epochs=max_epochs,
            steps_per_epoch=train_steps,
            validation_data=test_ds,
            validation_steps=val_steps,
            callbacks=[early_stopping],
            verbose=1
        )        # Test predictions - create a fresh dataset for testing
        model_manager.save_model()
        
        tf.random.set_seed(RANDOM_SEED)  # Reset seed for consistent test sampling

        sample_test_ds = data_manager.get_dataset().batch(BATCH_SIZE)  # Take 50 samples for testing
        sample_batch = next(iter(sample_test_ds))
        features, labels = sample_batch
        predictions = model(features)
        
        print("=== WIDE & DEEP WEIGHT ===")
        print(f"Wide weight: {model.wide_weight.numpy():.4f}")
        print(f"Deep weight: {model.deep_weight.numpy():.4f}")

        # Enhanced analysis for engagement score
        print("=== ENGAGEMENT SCORE ANALYSIS ===")
        print(f"Sample predictions range: {predictions.numpy().min():.4f} - {predictions.numpy().max():.4f}")
        print(f"Sample predictions mean: {predictions.numpy().mean():.4f}")
        print(f"Sample predictions std: {predictions.numpy().std():.4f}")
        print(f"Sample predictions: {predictions.numpy().flatten()[:3]}")  # Show first 3

        # Check label distribution
        sample_labels = labels.numpy().flatten()
        print(f"Sample labels range: {sample_labels.min():.4f} - {sample_labels.max():.4f}")
        print(f"Sample labels mean: {sample_labels.mean():.4f}")
        print(f"Sample labels std: {sample_labels.std():.4f}")

        # Check prediction vs label correlation
        correlation = np.corrcoef(predictions.numpy().flatten(), sample_labels)[0,1] if len(sample_labels) > 1 else 0.0
        print(f"Prediction-Label correlation: {correlation:.4f}")
        
        # Calculate R² score manually for verification
        def calculate_r2_score(y_true, y_pred):
            """Calculate R² score manually"""
            y_true_mean = np.mean(y_true)
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - y_true_mean) ** 2)
            return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        r2_score = calculate_r2_score(sample_labels, predictions.numpy().flatten())
        print(f"R² Score: {r2_score:.4f}")
        
        # Calculate prediction quality metrics
        prediction_variance = float(predictions.numpy().var())
        label_variance = float(sample_labels.var())
        mean_absolute_difference = float(np.mean(np.abs(predictions.numpy().flatten() - sample_labels)))
        
        print(f"Prediction variance: {prediction_variance:.4f}")
        print(f"Label variance: {label_variance:.4f}")
        print(f"Mean absolute difference: {mean_absolute_difference:.4f}")
        
        # Save new model

        return {
            "status": "success",
            "train_size": int(train_size),
            "test_size": int(test_size),
            "train_steps": int(train_steps),
            "val_steps": int(val_steps),
            "final_loss": float(history.history['loss'][-1]),
            #"final_val_loss": float(history.history.get('val_loss', [0])[-1]) if 'val_loss' in history.history else None,
            "final_rmse": float(history.history.get('root_mean_squared_error', [0])[-1]) if 'root_mean_squared_error' in history.history else None,
            "final_mae": float(history.history.get('mean_absolute_error', [0])[-1]) if 'mean_absolute_error' in history.history else None,
            "final_r2_score": float(history.history.get('r2_score', [0])[-1]) if 'r2_score' in history.history else None,
            "prediction_range": f"{predictions.numpy().min():.4f} - {predictions.numpy().max():.4f}",
            "label_range": f"{sample_labels.min():.4f} - {sample_labels.max():.4f}",
            "correlation": float(correlation) if 'correlation' in locals() else None,
            "r2_score_manual": float(r2_score) if 'r2_score' in locals() else None,
            "epochs_trained": int(len(history.history['loss'])),
            "training_mode": "incremental" if incremental_training else "from_scratch",
            "learning_rate": float(learning_rate),
            "wide_weight": float(model.wide_weight.numpy()),
            "deep_weight": float(model.deep_weight.numpy()),
            "wide_deep_ratio": float(model.wide_weight.numpy() / (model.wide_weight.numpy() + model.deep_weight.numpy())),
            
            # Enhanced prediction quality metrics for comparison
            "prediction_variance": float(prediction_variance) if 'prediction_variance' in locals() else None,
            "label_variance": float(label_variance) if 'label_variance' in locals() else None,
            "mean_absolute_difference": float(mean_absolute_difference) if 'mean_absolute_difference' in locals() else None,
            "prediction_mean": float(predictions.numpy().mean()),
            "prediction_std": float(predictions.numpy().std()),
            "label_mean": float(sample_labels.mean()),
            "label_std": float(sample_labels.std()),
            
            # Model architecture info for comparison
            "model_architecture": "wide_deep_baseline",
            "attention_analysis": None,  # No attention in baseline model
            
            # Training performance indicators
            "training_efficiency": {
                "final_loss_improvement": float(history.history['loss'][0] - history.history['loss'][-1]) if len(history.history['loss']) > 1 else 0.0,
                "best_epoch": int(np.argmin(history.history['loss'])) + 1,
                "convergence_rate": float(np.mean(np.diff(history.history['loss']))) if len(history.history['loss']) > 1 else 0.0
            },
            
            "note": "Baseline Wide & Deep model (without attention) for comparison"
        }

    except Exception as e:
        print(f"Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": str(e)}
