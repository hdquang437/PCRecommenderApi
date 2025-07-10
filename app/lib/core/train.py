import os
import shutil
import warnings
import random

import pandas as pd

from app.lib.core import data_manager

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

def leave_last_one_out_split(data):
    """
    Split data using Leave-Last-One-Out strategy
    - For each user, take the last interaction as test
    - All previous interactions as train
    """
    print("🔄 Applying Leave-Last-One-Out splitting...")
    
    # Sort by user_id and timestamp (if available) or by index
    data_sorted = data.sort_values(['user_id', 'click_times'], ascending=[True, False])
    
    train_data = []
    test_data = []
    
    # Group by user
    for user_id, user_group in data_sorted.groupby('user_id'):
        user_interactions = user_group.copy()
        
        if len(user_interactions) >= 2:
            # Last interaction for test
            test_interaction = user_interactions.iloc[0:1]  # Most recent (highest click_times)
            train_interactions = user_interactions.iloc[1:]  # All previous
            
            train_data.append(train_interactions)
            test_data.append(test_interaction)
        else:
            # If user has only 1 interaction, put in train
            train_data.append(user_interactions)
    
    train_df = pd.concat(train_data, ignore_index=True) if train_data else pd.DataFrame()
    test_df = pd.concat(test_data, ignore_index=True) if test_data else pd.DataFrame()
    
    print(f"📊 Leave-Last-One-Out Results:")
    print(f"  - Users with ≥2 interactions: {len(test_data)}")
    print(f"  - Train samples: {len(train_df)}")
    print(f"  - Test samples: {len(test_df)}")
    print(f"  - Test coverage: {len(test_df) / len(data) * 100:.1f}% of total data")
    
    return train_df, test_df

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
        
        # Calculate class weights
        label_counts = data['label'].value_counts()
        print(f"Label counts: {label_counts}")

        # Apply Leave-Last-One-Out splitting
        train_data, test_data = leave_last_one_out_split(data)

        train_size = len(train_data)
        test_size = len(test_data)
        print(f"Train size: {train_size}, Test size: {test_size}")

        train_ds = data_manager.create_dataset_from_dataframe(train_data)
        test_ds = data_manager.create_dataset_from_dataframe(test_data)

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
                
                # RESET GATE NETWORK to prevent weight collapse
                if hasattr(model, 'reset_gate_network'):
                    model.reset_gate_network()
                    print("🔄 Gate network reset for fresh incremental training")
                
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
        
        learning_rate = 0.0002 if incremental_training else 0.0001  # Reduced for stability
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            run_eagerly=False
        )
        
        print(f"Training mode: {'Incremental' if incremental_training else 'From scratch'}")
        print(f"Learning rate: {learning_rate}")

        # Train model with early stopping for better continuous learning
        # Use adaptive training parameters based on mode - REDUCED for overfitting prevention
        max_epochs = 2 if incremental_training else 5  # Reduced from 3/7
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
        )
        model_manager.save_model()
        
        # Save new model    

        return {
            "status": "success",
            "train_size": int(train_size),
            "test_size": int(test_size),
            "train_steps": int(train_steps),
            "val_steps": int(val_steps),
            "final_loss": float(history.history['loss'][-1]),
            
            # Binary Classification Metrics
            "final_accuracy": float(history.history.get('accuracy', [0])[-1]) if 'accuracy' in history.history else None,
            "final_precision": float(history.history.get('precision', [0])[-1]) if 'precision' in history.history else None,
            "final_recall": float(history.history.get('recall', [0])[-1]) if 'recall' in history.history else None,
            "final_auc": float(history.history.get('auc', [0])[-1]) if 'auc' in history.history else None,
            "final_f1_score": float(history.history.get('f1_score', [0])[-1]) if 'f1_score' in history.history else None,
            
            # Validation metrics - NOW AVAILABLE!
            "final_val_loss": float(history.history.get('val_loss', [0])[-1]) if 'val_loss' in history.history else None,
            "final_val_accuracy": float(history.history.get('val_accuracy', [0])[-1]) if 'val_accuracy' in history.history else None,
            "final_val_precision": float(history.history.get('val_precision', [0])[-1]) if 'val_precision' in history.history else None,
            "final_val_recall": float(history.history.get('val_recall', [0])[-1]) if 'val_recall' in history.history else None,
            "final_val_auc": float(history.history.get('val_auc', [0])[-1]) if 'val_auc' in history.history else None,
            "final_val_f1_score": float(history.history.get('val_f1_score', [0])[-1]) if 'val_f1_score' in history.history else None,
            
            # Overfitting check
            "overfitting_check": {
                "train_val_loss_diff": float(history.history['loss'][-1] - history.history.get('val_loss', [0])[-1]) if 'val_loss' in history.history else None,
                "train_val_acc_diff": float(history.history.get('accuracy', [0])[-1] - history.history.get('val_accuracy', [0])[-1]) if 'val_accuracy' in history.history else None
            },

            "epochs_trained": int(len(history.history['loss'])),
            "training_mode": "incremental" if incremental_training else "from_scratch",
            "learning_rate": float(learning_rate),
                        
            # Debug: Print available keys
            "available_metrics": list(history.history.keys()),

            # Model architecture info for comparison
            "model_architecture": "wide_deep_dynamic_weighting_binary_classification",
            "task_type": "binary_classification",
            "target_variable": "buy_prediction",
            
            # Training performance indicators
            "training_efficiency": {
                "final_loss_improvement": float(history.history['loss'][0] - history.history['loss'][-1]) if len(history.history['loss']) > 1 else 0.0,
                "best_epoch_loss": int(np.argmin(history.history['loss'])) + 1,
                "best_epoch_accuracy": int(np.argmax(history.history.get('accuracy', [0]))) + 1 if 'accuracy' in history.history else None,
                "convergence_rate": float(np.mean(np.diff(history.history['loss']))) if len(history.history['loss']) > 1 else 0.0,
                "accuracy_improvement": float(history.history.get('accuracy', [0])[-1] - history.history.get('accuracy', [0])[0]) if 'accuracy' in history.history and len(history.history['accuracy']) > 1 else 0.0
            },
            
            "note": "Wide & Deep model Baseline for Binary Purchase Prediction"
        }

    except Exception as e:
        print(f"Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": str(e)}