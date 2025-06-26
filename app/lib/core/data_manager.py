import asyncio
import json
import os
import threading
from datetime import datetime
import firebase_admin
from firebase_admin import credentials
import pandas as pd
import tensorflow as tf
from ...lib.models.interaction_repository import InteractionRepository
from ...lib.models.item_repository import ItemRepository
from ...lib.models.user_repository import UserRepository
from ...lib.models.shop_repository import ShopRepository
from ...paths import FIREBASE_KEY_PATH

# CONSTANTS cho stream data optimization
RELOAD_DEBOUNCE_DELAY = 5.0  # Số giây chờ trước khi reload (có thể điều chỉnh)
MAX_RELOAD_DELAY = 60.0     # Thời gian tối đa chờ trước khi buộc phải reload
IGNORE_CHANGES_DURING_RELOAD = True  # Bỏ qua các thay đổi trong quá trình reload

class DataManager:
    _instance = None  # Biến lưu instance duy nhất của class
    _lock = threading.Lock()  # Lock cho thread safety

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:  # Thread-safe singleton
                if cls._instance is None:  # Double-check locking
                    cls._instance = super(DataManager, cls).__new__(cls)
                    cls._instance.data = None
                    cls._instance.dataset = None
                    cls._instance.firebaseData = {}
                    cls._instance.listeners = []
                    cls._instance.is_local_deployment = False  # Flag để kiểm tra local deployment
                    cls._instance.reload_timer = None  # Timer cho debouncing
                    cls._instance.reload_pending = False  # Flag để kiểm tra reload đang pending
                    cls._instance.is_reloading = False  # Flag để kiểm tra đang reload
                    cls._instance.first_reload_time = None  # Thời gian reload đầu tiên được request
                    cls._instance.changes_ignored_count = 0  # Số lượng changes bị ignore
                    cls._instance.data_lock = threading.RLock()  # Lock cho data operations
                    cls._instance.init_lock = threading.Lock()  # Lock cho initialization
                    cls._instance.is_initializing = False  # Flag cho initialization
                    
                    # Kiểm tra nếu chưa được khởi tạo
                    if not firebase_admin._apps:
                        try:
                            # Thử sử dụng file Firebase key trước
                            cred = credentials.Certificate(FIREBASE_KEY_PATH)
                            firebase_admin.initialize_app(cred)
                            cls._instance.is_local_deployment = True  # Đang chạy local
                            print(f"Firebase initialized with key file: {FIREBASE_KEY_PATH}")
                        except Exception as e:
                            print(f"Failed to load Firebase key from file: {e}")
                            try:
                                # Fallback: sử dụng environment variable
                                firebase_cred_json = os.getenv("FIREBASE_KEY_JSON")
                                if firebase_cred_json:
                                    cred_dict = json.loads(firebase_cred_json)
                                    cred = credentials.Certificate(cred_dict)
                                    firebase_admin.initialize_app(cred)
                                    cls._instance.is_local_deployment = False  # Đang chạy production
                                    print("Firebase initialized with environment variable FIREBASE_KEY_JSON")
                                else:
                                    raise ValueError("Both FIREBASE_KEY_PATH file and FIREBASE_KEY_JSON environment variable are unavailable")
                            except Exception as env_error:
                                raise ValueError(f"Failed to initialize Firebase: {env_error}")
                    
                    cls.loop = asyncio.get_event_loop()

        return cls._instance

    def start_streams(self):
        """Bắt đầu lắng nghe thay đổi từ Firestore với debounced reload"""
        print("Starting Firestore stream listeners...")
        
        # INITIAL LOAD: Load data ngay lập tức khi khởi tạo
        print("Performing initial data load...")
        asyncio.create_task(self._handle_initial_load())
        
        user_repo = UserRepository()
        item_repo = ItemRepository()
        interaction_repo = InteractionRepository()
        shop_repo = ShopRepository()

        # Tạo wrapper function để merge tất cả các changes với optimized debouncing
        def merged_change_handler(collection_name):
            def handler(docs, changes, read_time):
                print(f"Change detected in {collection_name} collection")
                self._schedule_optimized_reload(collection_name)
            return handler

        self.listeners.append(user_repo.listen(merged_change_handler("users")))
        self.listeners.append(item_repo.listen(merged_change_handler("items")))
        self.listeners.append(interaction_repo.listen(merged_change_handler("interactions")))
        self.listeners.append(shop_repo.listen(merged_change_handler("shops")))

    def _schedule_optimized_reload(self, collection_name):
        """Optimized reload scheduling với advanced debouncing"""
        current_time = datetime.now()
        
        # Nếu đang reload, ignore change này
        if self.is_reloading and IGNORE_CHANGES_DURING_RELOAD:
            self.changes_ignored_count += 1
            print(f"Change ignored during reload (collection: {collection_name}, ignored: {self.changes_ignored_count})")
            return
        
        # Nếu chưa có reload nào được schedule
        if not self.reload_pending:
            self.reload_pending = True
            self.first_reload_time = current_time
            self.changes_ignored_count = 0
            print(f"First change detected in {collection_name}, scheduling reload in {RELOAD_DEBOUNCE_DELAY}s...")
        else:
            # Đã có reload pending, kiểm tra thời gian
            time_since_first = (current_time - self.first_reload_time).total_seconds()
            
            if time_since_first >= MAX_RELOAD_DELAY:
                # Đã chờ quá lâu, reload ngay lập tức
                print(f"Max delay reached ({MAX_RELOAD_DELAY}s), forcing reload now...")
                self._force_execute_reload()
                return
            else:
                # Vẫn trong thời gian chờ, reset timer
                print(f"Additional change in {collection_name}, resetting timer (total wait: {time_since_first:.1f}s)")
        
        # Hủy timer cũ và tạo timer mới
        if self.reload_timer:
            self.reload_timer.cancel()
        
        self.reload_timer = threading.Timer(RELOAD_DEBOUNCE_DELAY, self._execute_reload)
        self.reload_timer.start()

    def _force_execute_reload(self):
        """Buộc thực hiện reload ngay lập tức"""
        if self.reload_timer:
            self.reload_timer.cancel()
            self.reload_timer = None
        self._execute_reload()

    def _schedule_debounced_reload(self):
        """Legacy method - redirect to optimized version"""
        self._schedule_optimized_reload("unknown")
    def _execute_reload(self):
        """Thực hiện reload data với optimization"""
        if self.reload_pending:
            print(f"Executing optimized data reload... (ignored {self.changes_ignored_count} changes)")
            self.reload_pending = False
            self.reload_timer = None
            self.is_reloading = True  # Đánh dấu đang reload
            self.first_reload_time = None
            self.loop.call_soon_threadsafe(asyncio.create_task, self._handle_reload())

    def _on_change(self, docs, changes, read_time):
        """Legacy method - kept for compatibility"""
        print("Change detected, reloading data...")
        self.loop.call_soon_threadsafe(asyncio.create_task, self._handle_reload())

    async def _handle_reload(self):
        try:
            print("Starting data reload...")
            await self.load_data(reload=True)
            self.preprocess_data(reload=True)
            print("Data reload completed successfully!")
        except Exception as e:
            print(f"Error during data reload: {str(e)}")
        finally:
            # CRITICAL: Reset is_reloading flag để cho phép reload tiếp theo
            self.is_reloading = False
            print("Reload flag reset - ready for next changes")

    async def _handle_initial_load(self):
        """Load data ngay lập tức khi khởi tạo, không cần chờ changes"""
        try:
            print("Starting initial data load...")
            self.is_reloading = True  # Set flag during initial load
            await self.load_data(reload=True)
            self.preprocess_data(reload=True)
            print("Initial data load completed successfully!")
        except Exception as e:
            print(f"Error during initial data load: {str(e)}")
        finally:
            # Reset flag sau khi initial load hoàn thành
            self.is_reloading = False
            print("Initial load completed - ready for change detection")

    async def load_data(self, reload=False):
        """Load dữ liệu từ đầu nếu reload=True, nếu không dùng dữ liệu cũ."""
        if self.data is None or reload:
            # self.data = pd.DataFrame({
            #     "user_id": ["U001", "U001", "U002", "U002", "U003"],
            #     "gender": ["Male", "Male", "Female", "Female", "Male"],
            #     "age_range": ["18-25", "18-25", "26-35", "26-35", "36-50"],
            #     "product_id": ["P001", "P002", "P003", "P004", "P005"],
            #     "type": ["Clothing", "Electronics", "Food", "Furniture", "Shoes"],
            #     "price_range": ["Medium", "High", "Low", "High", "Medium"],
            #     "location": ["TP. HCM", "Hà Nội", "TP. HCM", "Hà Nội", "Đà Nẵng"],
            #     "click_times": [3, 0, 5, 2, 10],
            #     "buy_times": [1, 0, 2, 0, 1],
            #     "rating": [4.5, None, 4.0, None, 3.8],
            #     "label": [1.0, 0.0, 1.0, 0.0, 1.0]
            # })

            print("Loading dataset from Firestore...")
            
            user_repo = UserRepository()
            item_repo = ItemRepository()
            interaction_repo = InteractionRepository()
            shop_repo = ShopRepository()

            users = await user_repo.get_all_users()
            items = await item_repo.get_all_items()
            interactions = await interaction_repo.get_all_interactions()
            shops = await shop_repo.get_all_shops()

            user_df = pd.DataFrame([user.__dict__ for user in users])
            shop_df = pd.DataFrame([shop.__dict__ for shop in shops])
            item_df = pd.DataFrame([item.__dict__ for item in items])
            interaction_df = pd.DataFrame([interaction.__dict__ for interaction in interactions])
            # Lấy location người bán từ user_df
            seller_df = shop_df[["id", "location"]].rename(columns={"id": "seller_id", "location": "seller_location"})

            self.firebaseData["users"] = user_df
            self.firebaseData["items"] = item_df
            self.firebaseData["interactions"] = interaction_df
            self.firebaseData["sellers"] = seller_df

            # Chuyển date_of_birth thành age_range
            current_year = datetime.now().year

            user_df["age_range"] = user_df["date_of_birth"].apply(
                lambda dob:
                    "18-25" if current_year - dob.year <= 25
                        else "26-35" if current_year - dob.year <= 35
                        else "36-50" if current_year - dob.year <= 50
                        else "50+"
            )
            
            # Chuyển price thành price_range
            item_df["price_range"] = item_df["price"].apply(
                lambda p:
                    "budget" if p <= 200000
                        else "mid-range" if p <= 1000000
                        else "upper mid-range" if p <= 3000000
                        else "premium" if p <= 7000000
                        else "flagship"
            )
            
            # Gán location vào item theo seller_id
            item_df = item_df.merge(seller_df, on="seller_id", how="left")

            # Merge dữ liệu dựa trên user_id và item_id
            self.data = interaction_df.merge(user_df, left_on="user_id", right_on="id", how="left")
            self.data = self.data.merge(item_df, left_on="item_id", right_on="id", how="left")
            
            # Chọn cột đúng format
            self.data = self.data[[
                "user_id", "gender", "age_range", "item_id", "item_type", "price_range", "seller_location", "click_times", "buy_times", "rating"
            ]]
            
            # Đổi tên cột để khớp với format cũ
            self.data.rename(columns={
                "item_id": "product_id",
                "item_type": "type",
                "seller_location": "location",            }, inplace=True)
            
            # Tạo label
            self.data["label"] = self.data["click_times"].apply(lambda x: 1.0 if x > 0 else 0.0)
            
            self.data.fillna({"click_times": 0, "buy_times": 0, "rating": 3.0}, inplace=True)
            
            # Export Firebase data to CSV files for analysis (chỉ khi local deployment)
            if self.is_local_deployment:
                self._export_firebase_data_to_csv()
                print("CSV export completed for local deployment")
            else:
                print("Skipping CSV export - running in production mode")
            
            print("Data loaded!")
            # print(self.data)

    def preprocess_data(self, reload=False):
        """Tiền xử lý dữ liệu thành TensorFlow dataset."""
        if self.dataset is None or reload:
            print("Preprocessing dataset...")

            # Load category mappings từ file JSON
            self._load_category_mappings_from_file()

            # Chuyển đổi categorical thành số sử dụng category_mappings từ file
            for field in ["gender", "age_range", "type", "price_range", "location"]:
                if field in self.category_mappings:
                    categories = self.category_mappings[field]
                    # Tạo categorical với tất cả categories từ mapping file
                    self.data[field] = pd.Categorical(self.data[field], categories=categories)
                    self.data[field] = self.data[field].cat.codes
                else:
                    # Fallback nếu không có trong mapping file
                    self.data[field] = self.data[field].astype("category").cat.codes

            self.data["click_times"] = self.data["click_times"].astype(float)
            self.data["buy_times"] = self.data["buy_times"].astype(float)
            self.data["rating"] = self.data["rating"].astype(float)
            self.data["label"] = self.data["label"].astype(float)

            # Tạo TensorFlow dataset

            self.dataset = tf.data.Dataset.from_tensor_slices((
                dict(self.data.drop(columns=["user_id", "product_id"])),
                self.data["label"]
            ))

            self.dataset = self.dataset.map(lambda x, y: (
                {k: tf.cast(v, tf.float32) if v.dtype == tf.float64 else tf.cast(v, tf.int32) for k, v in x.items()},
                tf.cast(y, tf.float32)
            ))

            self.dataset = self.dataset.shuffle(len(self.data)).batch(2, drop_remainder=True)
            
            print("Data preprocessing completed!")

    def get_data(self):
        """Trả về dữ liệu dưới dạng pandas DataFrame - Thread safe read."""
        with self.data_lock:
            if self.data is None:
                raise ValueError("Data not loaded yet")
            # Trả về copy để tránh modification từ bên ngoài
            return self.data.copy()

    def get_dataset(self):
        """Trả về dữ liệu dưới dạng TensorFlow dataset - Thread safe read."""
        with self.data_lock:
            if self.dataset is None:
                raise ValueError("Dataset not preprocessed yet")
            return self.dataset
    
    def get_product_ids(self):
        """Return product ids - Thread safe."""
        with self.data_lock:
            if self.data is None:
                raise ValueError("Data not loaded yet")
            return self.data["product_id"].unique().tolist()

    def build_empty_sample(self, user_id, product_id):
        """Tạo sample giả với buy_times, click_times, rating = 0 nếu chưa có tương tác - Thread safe."""
        # Thread-safe access to firebase data
        with self.data_lock:
            user_df = self.firebaseData.get("users")
            item_df = self.firebaseData.get("items")
            seller_df = self.firebaseData.get("sellers")
            
            if user_df is None or item_df is None or seller_df is None:
                raise ValueError("Firebase data not loaded properly")

            # Lấy thông tin user
            user_row = user_df[user_df["id"] == user_id]
            if user_row.empty:
                raise ValueError(f"User ID {user_id} not found")
            user_row = user_row.iloc[0]

            # Lấy thông tin item
            item_row = item_df[item_df["id"] == product_id]
            if item_row.empty:
                raise ValueError(f"Product ID {product_id} not found")
            item_row = item_row.iloc[0]

            # Lấy location từ seller
            seller_row = seller_df[seller_df["seller_id"] == item_row["seller_id"]]
            location = seller_row["seller_location"].values[0] if not seller_row.empty else "unknown"

        # Tính age_range
        current_year = datetime.now().year
        age = current_year - user_row["date_of_birth"].year
        if age <= 25:
            age_range = "18-25"
        elif age <= 35:
            age_range = "26-35"
        elif age <= 50:
            age_range = "36-50"
        else:
            age_range = "50+"

        # Tính price_range
        price = item_row["price"]
        if price <= 200000:
            price_range = "budget"
        elif price <= 1000000:
            price_range = "mid-range"
        elif price <= 3000000:
            price_range = "upper mid-range"
        elif price <= 7000000:
            price_range = "premium"
        else:
            price_range = "flagship"

        sample = {
            "user_id": user_id,
            "gender": user_row["gender"],
            "age_range": age_range,
            "product_id": product_id,
            "type": item_row["item_type"],
            "price_range": price_range,
            "location": location,
            "click_times": 0.0,
            "buy_times": 0.0,
            "rating": 0.0,
            "label": 0.0
        }

        return sample
    
    def encode_sample(self, sample_dict):
        """
        Chuyển đổi các trường categorical trong sample_dict thành số theo category_mappings.
        category_mappings: dict, key = field name, value = list các category theo thứ tự mã hóa.
        """
        sample_encoded = sample_dict.copy()
        category_mappings = self.category_mappings

        for col in ["gender", "age_range", "type", "price_range", "location"]:
            categories = category_mappings.get(col)
            if categories is None:
                raise ValueError(f"Missing category mapping for {col}")

            val = sample_dict[col]
            if val in categories:
                sample_encoded[col] = categories.index(val)
            else:
                # Gán 0 cho unknown category (thay vì -1 để tránh out-of-bounds)
                print(f"Warning: Unknown category '{val}' for field '{col}', using default value 0")
                sample_encoded[col] = 0        # Đảm bảo các trường số là float
        sample_encoded["click_times"] = float(sample_encoded["click_times"])
        sample_encoded["buy_times"] = float(sample_encoded["buy_times"])
        sample_encoded["rating"] = float(sample_encoded["rating"])
        
        return sample_encoded
    
    def get_vocab_sizes(self):
        """Trả về kích thước vocabulary cho từng categorical field"""
        if not hasattr(self, 'category_mappings') or self.category_mappings is None:
            raise ValueError("Category mappings not initialized. Please call preprocess_data() first.")
        
        vocab_sizes = {}
        for field, categories in self.category_mappings.items():
            vocab_sizes[field] = len(categories)
        
        # DEBUG: Print actual ranges để debug index out of bounds
        if self.data is not None:
            print(f"DEBUG - Vocab sizes: {vocab_sizes}")
            for field in ["type", "location", "gender", "age_range", "price_range"]:
                if field in self.data.columns:
                    field_min = self.data[field].min()
                    field_max = self.data[field].max()
                    print(f"DEBUG - {field} range: {field_min} - {field_max} (vocab_size: {vocab_sizes.get(field, 'N/A')})")
                    
                    # Check for potential out-of-bounds
                    if field in vocab_sizes and field_max >= vocab_sizes[field]:
                        print(f"WARNING - {field} max value {field_max} >= vocab_size {vocab_sizes[field]}!")
        
        return vocab_sizes
    
    def _export_firebase_data_to_csv(self):
        """Export Firebase data to CSV files for easier analysis and debugging"""
        try:
            import os
            
            # Create export directory if it doesn't exist
            export_dir = os.path.join(os.path.dirname(__file__), '..', 'firebaseDataLog')
            os.makedirs(export_dir, exist_ok=True)
            
            # Export individual Firebase collections (override existing files)
            if "users" in self.firebaseData and not self.firebaseData["users"].empty:
                users_file = os.path.join(export_dir, "users.csv")
                self.firebaseData["users"].to_csv(users_file, index=False)
                print(f"Exported users data to: {users_file}")
            
            if "items" in self.firebaseData and not self.firebaseData["items"].empty:
                items_file = os.path.join(export_dir, "items.csv")
                self.firebaseData["items"].to_csv(items_file, index=False)
                print(f"Exported items data to: {items_file}")
            
            if "interactions" in self.firebaseData and not self.firebaseData["interactions"].empty:
                interactions_file = os.path.join(export_dir, "interactions.csv")
                self.firebaseData["interactions"].to_csv(interactions_file, index=False)
                print(f"Exported interactions data to: {interactions_file}")
            
            if "sellers" in self.firebaseData and not self.firebaseData["sellers"].empty:
                sellers_file = os.path.join(export_dir, "sellers.csv")
                self.firebaseData["sellers"].to_csv(sellers_file, index=False)
                print(f"Exported sellers data to: {sellers_file}")
              # Export the final processed dataset
            if self.data is not None and not self.data.empty:
                final_data_file = os.path.join(export_dir, "final_dataset.csv")
                self.data.to_csv(final_data_file, index=False)
                print(f"Exported final processed dataset to: {final_data_file}")
                
            print(f"All Firebase data exported to: {export_dir}")
            
        except Exception as e:
            print(f"Error exporting Firebase data to CSV: {str(e)}")
    
    def _load_category_mappings_from_file(self):
        """Load category mappings từ file JSON"""
        try:
            import json
            import os
            
            mappings_file = os.path.join(os.path.dirname(__file__), '..', 'core', 'category_mappings.json')
            
            if os.path.exists(mappings_file):
                with open(mappings_file, 'r', encoding='utf-8') as f:
                    self.category_mappings = json.load(f)
                print(f"Loaded category mappings from: {mappings_file}")
                print(f"Available categories: {list(self.category_mappings.keys())}")
                for field, categories in self.category_mappings.items():
                    print(f"  {field}: {len(categories)} categories")
            else:
                print(f"Category mappings file not found: {mappings_file}")
                # Fallback: tạo mappings từ data hiện tại
                self.category_mappings = {
                    "gender": list(self.data["gender"].astype("category").cat.categories),
                    "age_range": list(self.data["age_range"].astype("category").cat.categories),
                    "type": list(self.data["type"].astype("category").cat.categories),
                    "price_range": list(self.data["price_range"].astype("category").cat.categories),
                    "location": list(self.data["location"].astype("category").cat.categories),
                }
                
        except Exception as e:
            print(f"Error loading category mappings: {str(e)}")
            # Fallback: tạo mappings từ data hiện tại
            self.category_mappings = {
                "gender": list(self.data["gender"].astype("category").cat.categories),
                "age_range": list(self.data["age_range"].astype("category").cat.categories),
                "type": list(self.data["type"].astype("category").cat.categories),
                "price_range": list(self.data["price_range"].astype("category").cat.categories),
                "location": list(self.data["location"].astype("category").cat.categories),
            }