import threading
import time
from fastapi import FastAPI
import numpy as np
import pandas as pd
import tensorflow as tf
import os

from .lib.core.predict import predict, get_top_popular_products
from .lib.core.train import train_model
from .lib.core.data_manager import DataManager
from .paths import MODEL_PATH, CACHE_DELETED_ITEM_PATH, CACHE_USER_PATH, CACHE_ITEM_PATH, CACHE_INTERACTION_PATH

app = FastAPI()
data_manager = DataManager()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/recommend")
async def recommend(uid: str, max: int):
    try:
        # Đảm bảo data đã được load
        if data_manager.data is None:
            await data_manager.load_data()
            data_manager.preprocess_data()
            
        recommendations = []
        
        product_ids = data_manager.get_product_ids()
        top_k = min(max, len(product_ids))

        for product_id in product_ids:
            try:
                score = predict(uid, product_id)
                if score >= 0:  # Chỉ lấy những dự đoán hợp lệ
                    recommendations.append((product_id, score))
            except Exception as e:
                print(f"Error predicting for product {product_id}: {e}")
                continue
        
        # Sắp xếp theo xác suất cao nhất
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        # Lấy top k sản phẩm
        top_products = [prod[0] for prod in recommendations[:top_k]]
        
        # Nếu chưa đủ k sản phẩm, lấy các sản phẩm phổ biến nhất
        if len(top_products) < top_k:
            try:
                popular_products = get_top_popular_products(top_k - len(top_products))
                # Loại bỏ duplicate
                for prod in popular_products:
                    if prod not in top_products and len(top_products) < top_k:
                        top_products.append(prod)
            except Exception as e:
                print(f"Error getting popular products: {e}")

        return {"products": top_products}
    except Exception as e:
        print(f"Error in recommend endpoint: {e}")
        return {"error": str(e), "products": []}

@app.post("/build")
async def build():
    print("Building!")
    try:
        # Đảm bảo data đã được load trước khi train
        if data_manager.data is None:
            await data_manager.load_data()
            data_manager.preprocess_data()
            
        train_model()
        print("Build done!")
        return {"message": "Build successfully"}
    except Exception as e:
        print("Build failed with error: ")
        print(e)
        return {"message": str(e), "error": True}


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
async def init():
    try:
        print("Starting up application...")
        await data_manager.load_data(True)
        data_manager.preprocess_data()
        data_manager.start_streams()
        os.environ["TF_DATA_AUTOTUNE_RAM_budget"] = "104857600"  # 100 MB
        print("Application startup completed!")
    except Exception as e:
        print(f"Error during startup: {e}")
        # Không raise exception để app vẫn có thể start, nhưng log lỗi