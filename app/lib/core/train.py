import tensorflow as tf
from .data_manager import DataManager
from .model_manager import ModelManager
from tensorflow import keras
from keras import layers

def train_model():
  BATCH_SIZE = 32
  TRAIN_RATIO = 0.8  # 80% train, 20% test

  # Khởi tạo DataManager (Singleton)
  data_manager = DataManager()

  # Lấy dataset
  data = data_manager.get_data()
  dataset = data_manager.get_dataset()

  print("Data columns:", dataset)

  # Chia train / test
  train_size = int(TRAIN_RATIO * len(data))
  train_ds = dataset.take(train_size)
  test_ds = dataset.skip(train_size)
  # Khởi tạo ModelManager
  model_manager = ModelManager()
  vocab_sizes = data_manager.get_vocab_sizes()
  model = model_manager.load_model(vocab_sizes=vocab_sizes, reload=True)

  # Tự động điều chỉnh repeat & steps_per_epoch
  if train_size > 10000:  # Ngưỡng để quyết định dataset lớn
    steps_per_epoch = None  # Keras sẽ tự tính nếu None
  else:
    train_ds = train_ds.repeat()
    test_ds = test_ds.repeat()
    steps_per_epoch = max(1, train_size // BATCH_SIZE)
  # Train mô hình
  model.fit(train_ds, epochs=10, validation_data=test_ds, steps_per_epoch=steps_per_epoch)

  # Gọi model trên một batch dữ liệu test trước khi save
  dummy_input = next(iter(train_ds.take(1)))[0]  # Lấy features
  model(dummy_input)

  # Lưu mô hình
  model_manager.save_model()
