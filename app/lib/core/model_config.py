import os

# Model Configuration
BATCH_SIZE = 8
RANDOM_SEED = 42

# Data Source Configuration 
USE_CSV = os.getenv("USE_CSV_DATA", "true").lower() == "true"
CSV_DATA_PATH = os.getenv("CSV_DATA_PATH", "app/lib/samples")
CSV_FILENAME = os.getenv("CSV_FILENAME", "test_mock_dataset.csv")

# Development override - Force CSV mode for development
DEVELOPMENT_MODE = True  # Set to False for production
if DEVELOPMENT_MODE:
    USE_CSV = True
    print(f"🔧 DEVELOPMENT MODE: Forcing CSV usage")
    print(f"📁 CSV File: {CSV_DATA_PATH}/{CSV_FILENAME}")

# Print current config for debugging
print(f"🔧 Model Config Loaded:")
print(f"  USE_CSV: {USE_CSV}")
print(f"  CSV_DATA_PATH: {CSV_DATA_PATH}")
print(f"  CSV_FILENAME: {CSV_FILENAME}")
print(f"  BATCH_SIZE: {BATCH_SIZE}")
print(f"  RANDOM_SEED: {RANDOM_SEED}")

# Optional: Environment-based overrides
if os.getenv("DEBUG_MODE", "false").lower() == "true":
    print(f"🐛 DEBUG MODE ENABLED")
    BATCH_SIZE = 16  # Smaller batch for debugging