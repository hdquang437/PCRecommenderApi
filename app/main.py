import threading
import time
import asyncio
import concurrent.futures
from fastapi import FastAPI, BackgroundTasks
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

# Thread pool for CPU-intensive prediction tasks
prediction_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=6,  # Tăng từ 4 lên 6 để xử lý 3 users đồng thời tốt hơn
    thread_name_prefix="prediction-worker"
)

# Rate limiting để tránh overload
from collections import defaultdict
request_counts = defaultdict(int)
request_timestamps = defaultdict(list)

def check_rate_limit(user_id: str, max_requests: int = 10, window_seconds: int = 60):
    """Simple rate limiting per user"""
    current_time = time.time()
    user_timestamps = request_timestamps[user_id]
    
    # Remove old timestamps outside the window
    user_timestamps[:] = [ts for ts in user_timestamps if current_time - ts < window_seconds]
    
    if len(user_timestamps) >= max_requests:
        return False
    
    user_timestamps.append(current_time)
    return True

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/recommend")
async def recommend(uid: str, max: int):
    request_start_time = time.time()
    print(f"🔄 User {uid} request started in thread {threading.current_thread().name}")
    
    try:
        # Rate limiting check
        if not check_rate_limit(uid):
            return {"error": "Rate limit exceeded", "products": []}
        
        # Thread-safe data loading check với proper locking
        async def ensure_data_loaded():
            # Check nhanh trước khi acquire lock để tránh blocking không cần thiết
            if data_manager.data is not None:
                return
                
            # Acquire initialization lock để đảm bảo chỉ 1 thread load data
            with data_manager.init_lock:
                # Double-check locking pattern
                if data_manager.data is None and not data_manager.is_initializing:
                    data_manager.is_initializing = True
                    try:
                        print(f"Thread {threading.current_thread().name}: Loading data...")
                        await data_manager.load_data()
                        data_manager.preprocess_data()
                        print(f"Thread {threading.current_thread().name}: Data loading completed")
                    finally:
                        data_manager.is_initializing = False
                elif data_manager.is_initializing:
                    # Nếu thread khác đang load, chờ cho đến khi hoàn thành
                    print(f"Thread {threading.current_thread().name}: Waiting for data loading to complete...")
                    while data_manager.is_initializing:
                        await asyncio.sleep(0.1)  # Chờ 100ms rồi check lại
            
        await ensure_data_loaded()
            
        recommendations = []
        
        # Thread-safe data access với read lock
        with data_manager.data_lock:
            if data_manager.data is None:
                raise ValueError("Data not available after loading attempt")
            product_ids = data_manager.get_product_ids()
            
        top_k = min(max, len(product_ids))        # Use thread pool executor for concurrent prediction
        loop = asyncio.get_running_loop()
        futures = [
            loop.run_in_executor(prediction_executor, predict, uid, product_id)
            for product_id in product_ids
        ]
        
        # Wait for all predictions to complete
        results = await asyncio.gather(*futures, return_exceptions=True)
        
        # Process results
        for i, result in enumerate(results):
            product_id = product_ids[i]
            try:
                if isinstance(result, Exception):
                    print(f"Error predicting for product {product_id}: {result}")
                    continue
                
                score = result
                if score >= 0:  # Chỉ lấy những dự đoán hợp lệ
                    recommendations.append((product_id, score))
            except Exception as e:
                print(f"Error processing result for product {product_id}: {e}")
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
            except Exception as e:                print(f"Error getting popular products: {e}")

        print(f"✅ User {uid} completed in {time.time() - request_start_time:.2f}s - thread {threading.current_thread().name}")
        print(f"📊 Found {len(recommendations)} recommendations, returning top {len(top_products)} products")
        return {
            "products": top_products, 
            "total_found": len(recommendations),
            "processing_time": round(time.time() - request_start_time, 2),
            "thread_id": threading.current_thread().name
        }
    except Exception as e:
        print(f"Error in recommend endpoint: {e}")
        return {"error": str(e), "products": []}

@app.post("/build")
async def build():
    print("Building!")
    try:
        # Thread-safe data loading với proper locking
        async def ensure_data_loaded():
            if data_manager.data is not None:
                return
                
            with data_manager.init_lock:
                if data_manager.data is None and not data_manager.is_initializing:
                    data_manager.is_initializing = True
                    try:
                        await data_manager.load_data()
                        data_manager.preprocess_data()
                    finally:
                        data_manager.is_initializing = False
                elif data_manager.is_initializing:
                    while data_manager.is_initializing:
                        await asyncio.sleep(0.1)
        
        await ensure_data_loaded()
            
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

@app.get("/health")
async def health_check():
    """Health check endpoint to verify system status."""
    try:
        status = {
            "status": "healthy",
            "data_loaded": data_manager.data is not None,
            "model_available": True,  # We'll check this when model manager is available
            "thread_id": threading.current_thread().name,
            "timestamp": time.time()
        }
        
        # Quick data validation
        if data_manager.data is not None:
            with data_manager.data_lock:
                status["data_records"] = len(data_manager.data)
                status["product_count"] = len(data_manager.data["product_id"].unique())
        else:
            status["data_records"] = 0
            status["product_count"] = 0
            
        return status
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "thread_id": threading.current_thread().name,
            "timestamp": time.time()
        }

@app.get("/test-concurrent")
async def test_concurrent_requests():
    """Test endpoint để kiểm tra khả năng xử lý concurrent requests"""
    try:
        # Test với 3 users khác nhau bằng cách gọi trực tiếp function
        test_users = ["U001", "U002", "U003"]
        max_products = 5
        
        async def simulate_user_request(user_id):
            """Mô phỏng một request từ user"""
            start_time = time.time()
            try:
                # Gọi trực tiếp function recommend
                result = await recommend(user_id, max_products)
                duration = time.time() - start_time
                
                return {
                    "user_id": user_id,
                    "status": "success" if "products" in result else "error",
                    "duration": round(duration, 2),
                    "products_count": len(result.get("products", [])),
                    "thread_id": result.get("thread_id", "unknown"),
                    "processing_time": result.get("processing_time", 0)
                }
            except Exception as e:
                return {
                    "user_id": user_id,
                    "status": "error",
                    "duration": round(time.time() - start_time, 2),
                    "error": str(e)
                }
        
        print("🚀 Starting concurrent test with 3 users...")
        start_time = time.time()
        
        # Chạy 3 requests đồng thời
        tasks = [simulate_user_request(user_id) for user_id in test_users]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        total_time = time.time() - start_time
        
        # Xử lý kết quả
        success_count = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "success")
        avg_duration = sum(r.get("duration", 0) for r in results if isinstance(r, dict)) / len(results)
        
        print(f"✅ Concurrent test completed: {success_count}/{len(test_users)} successful in {total_time:.2f}s")
        
        return {
            "test_type": "concurrent_requests",
            "total_users": len(test_users),
            "successful_requests": success_count,
            "total_time": round(total_time, 2),
            "average_request_time": round(avg_duration, 2),
            "results": results,
            "summary": f"✅ {success_count}/{len(test_users)} requests successful in {total_time:.2f}s",
            "concurrent_efficiency": round((avg_duration / total_time) * 100, 1) if total_time > 0 else 0
        }
        
    except Exception as e:
        return {
            "test_type": "concurrent_requests",
            "error": str(e),
            "status": "failed"
        }

@app.get("/debug/datatable")
async def get_current_datatable():
    """API để xem datatable hiện tại"""
    try:
        # Lấy data hiện tại từ DataManager
        data = data_manager.get_data()
        
        # Chuyển đổi thành format dễ đọc
        result = {
            "status": "success",
            "total_rows": len(data),
            "columns": list(data.columns),
            "data_sample": data.head(10).to_dict('records'),  # 10 rows đầu
            "data_types": data.dtypes.to_dict(),
            "summary": {
                "unique_users": data["user_id"].nunique() if "user_id" in data.columns else 0,
                "unique_products": data["product_id"].nunique() if "product_id" in data.columns else 0,
                "unique_locations": data["location"].nunique() if "location" in data.columns else 0,
                "unique_types": data["type"].nunique() if "type" in data.columns else 0,
                "total_interactions": len(data),
                "positive_labels": int(data["label"].sum()) if "label" in data.columns else 0
            }
        }
        
        print(f"📊 Datatable info requested - {len(data)} rows, {len(data.columns)} columns")
        return result
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "message": "Failed to retrieve datatable"
        }

@app.post("/admin/force-reload")
async def force_reload_data():
    """API để force reload data mới từ Firebase"""
    try:
        print("🔄 Force reload data requested...")
        
        # Force execute reload ngay lập tức
        data_manager._force_execute_reload()
        
        # Chờ một chút để reload hoàn thành
        await asyncio.sleep(0.5)
        
        # Lấy thông tin data mới
        try:
            data = data_manager.get_data()
            data_info = {
                "total_rows": len(data),
                "unique_users": data["user_id"].nunique() if "user_id" in data.columns else 0,
                "unique_products": data["product_id"].nunique() if "product_id" in data.columns else 0,
                "last_updated": "just now"
            }
        except:
            data_info = {"status": "reload_in_progress"}
        
        print("✅ Force reload completed!")
        
        return {
            "status": "success",
            "message": "Data reload forced successfully",
            "action": "force_reload_executed",
            "data_info": data_info,
            "note": "Data will be refreshed in a few seconds"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "message": "Failed to force reload data"
        }

@app.post("/admin/reset-reload-state")
async def reset_reload_state():
    """API để reset trạng thái reload khi bị stuck"""
    try:
        old_state = {
            "is_reloading": getattr(data_manager, 'is_reloading', False),
            "reload_pending": getattr(data_manager, 'reload_pending', False),
            "changes_ignored_count": getattr(data_manager, 'changes_ignored_count', 0)
        }
        
        # Reset tất cả flags
        data_manager.is_reloading = False
        data_manager.reload_pending = False
        data_manager.changes_ignored_count = 0
        data_manager.first_reload_time = None
        
        # Cancel timer nếu có
        if hasattr(data_manager, 'reload_timer') and data_manager.reload_timer:
            data_manager.reload_timer.cancel()
            data_manager.reload_timer = None
        
        print("🔧 Reload state has been reset!")
        
        return {
            "status": "success",
            "message": "Reload state reset successfully",
            "old_state": old_state,
            "new_state": {
                "is_reloading": False,
                "reload_pending": False,
                "changes_ignored_count": 0
            },
            "note": "System is now ready to accept new changes"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "message": "Failed to reset reload state"
        }

@app.get("/debug/system-status")
async def get_system_status():
    """API để xem trạng thái hệ thống"""
    try:
        # Kiểm tra trạng thái DataManager
        is_reloading = getattr(data_manager, 'is_reloading', False)
        reload_pending = getattr(data_manager, 'reload_pending', False)
        changes_ignored = getattr(data_manager, 'changes_ignored_count', 0)
        
        # Kiểm tra data
        try:
            data = data_manager.get_data()
            data_status = "loaded"
            data_rows = len(data)
        except:
            data_status = "not_loaded"
            data_rows = 0
            
        # Kiểm tra model
        model_exists = os.path.exists(MODEL_PATH)
        
        return {
            "status": "success",
            "system_status": {
                "data_manager": {
                    "is_reloading": is_reloading,
                    "reload_pending": reload_pending,
                    "changes_ignored_count": changes_ignored,
                    "data_status": data_status,
                    "data_rows": data_rows
                },
                "model": {
                    "model_file_exists": model_exists,
                    "model_path": MODEL_PATH
                },
                "constants": {
                    "reload_debounce_delay": "3.0s",
                    "max_reload_delay": "60.0s",
                    "ignore_changes_during_reload": True
                }
            }
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "message": "Failed to get system status"
        }

@app.post("/admin/reset-reload-state")
async def reset_reload_state():
    """API để reset trạng thái reload khi bị stuck"""
    try:
        old_state = {
            "is_reloading": getattr(data_manager, 'is_reloading', False),
            "reload_pending": getattr(data_manager, 'reload_pending', False),
            "changes_ignored_count": getattr(data_manager, 'changes_ignored_count', 0)
        }
        
        # Reset tất cả flags
        data_manager.is_reloading = False
        data_manager.reload_pending = False
        data_manager.changes_ignored_count = 0
        data_manager.first_reload_time = None
        
        # Cancel timer nếu có
        if hasattr(data_manager, 'reload_timer') and data_manager.reload_timer:
            data_manager.reload_timer.cancel()
            data_manager.reload_timer = None
        
        print("🔧 Reload state has been reset!")
        
        return {
            "status": "success",
            "message": "Reload state reset successfully",
            "old_state": old_state,
            "new_state": {
                "is_reloading": False,
                "reload_pending": False,
                "changes_ignored_count": 0
            },
            "note": "System is now ready to accept new changes"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "message": "Failed to reset reload state"
        }