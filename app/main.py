import threading
import time
from fastapi import FastAPI
import numpy as np
import pandas as pd
import tensorflow as tf
import os

from .lib.core.predict import predict
from .lib.core.train import train_model
from .paths import MODEL_PATH, CACHE_DELETED_ITEM_PATH, CACHE_USER_PATH, CACHE_ITEM_PATH, CACHE_INTERACTION_PATH

app = FastAPI()

@app.post("/recommend/")
async def recommend(uid: str):
    predict("U001", "P004")

# Hàm build lại model theo chu kỳ
def periodic_train_model(interval, model_rebuild_function):
    """
    Xây dựng lại mô hình theo chu kỳ.

    Args:
        interval (int): Chu kỳ tính theo giây để chờ xây dựng lại mô hình.
        model_rebuild_function (function): Hàm build lại mô hình
    """
    while True:
        # Ghi lại thời gian bắt đầu
        start_time = time.time()

        print("Rebuilding model...")
        model_rebuild_function()
        
        # Ghi lại thời gian kết thúc
        end_time = time.time()
        
        # Tính toán thời gian thực thi
        execution_time = end_time - start_time

        print(f"Model rebuild completed [{execution_time:.4f} seconds]. Waiting for next cycle...")
        
        time.sleep(interval)

@app.on_event("startup")
# Hàm khởi động thread xây dựng lại mô hình
async def start_background_model_rebuild():
    """
    Chạy background thread thực hiện xây dựng lại mô hình.
    """
    rebuild_interval = 1800  # 30 minutes
    rebuild_thread = threading.Thread(
        target=periodic_train_model,
        args=(rebuild_interval, train_model),
        daemon=True  # Ensures thread exits when main program exits
    )
    rebuild_thread.start()