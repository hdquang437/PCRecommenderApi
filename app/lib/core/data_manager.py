from datetime import datetime
import firebase_admin
from firebase_admin import credentials
import pandas as pd
import tensorflow as tf
from ...lib.models.interaction_repository import InteractionRepository
from ...lib.models.item_repository import ItemRepository
from ...lib.models.user_repository import UserRepository
from ...paths import FIREBASE_KEY_PATH

class DataManager:
    _instance = None  # Biến lưu instance duy nhất của class

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DataManager, cls).__new__(cls)
            cls._instance.data = None
            cls._instance.dataset = None
            cls._instance.listeners = []
            # Kiểm tra nếu chưa được khởi tạo
            if not firebase_admin._apps:
                cred = credentials.Certificate(FIREBASE_KEY_PATH)
                firebase_admin.initialize_app(cred)

            cls.start_streams
        return cls._instance

    def start_streams(self):
        """Bắt đầu lắng nghe thay đổi từ Firestore"""
        print("Starting Firestore stream listeners...")

        user_repo = UserRepository()
        item_repo = ItemRepository()
        interaction_repo = InteractionRepository()

        self.listeners.append(user_repo.listen(self._on_change))
        self.listeners.append(item_repo.listen(self._on_change))
        self.listeners.append(interaction_repo.listen(self._on_change))

    def _on_change(self, docs, changes, read_time):
        print("Change detected, reloading data...")
        self.load_data(reload=True)
        self.preprocess_data(reload=True)

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

            users = await user_repo.get_all_users()
            items = await item_repo.get_all_items()
            interactions = await interaction_repo.get_all_interactions()

            user_df = pd.DataFrame([user.__dict__ for user in users])
            item_df = pd.DataFrame([item.__dict__ for item in items])
            interaction_df = pd.DataFrame([interaction.__dict__ for interaction in interactions])
            # Lấy location người bán từ user_df
            seller_df = user_df[["id", "location"]].rename(columns={"id": "seller_id", "location": "seller_location"})

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
                "seller_location": "location",
            }, inplace=True)

            # Tạo label
            self.data["label"] = self.data["click_times"].apply(lambda x: 1.0 if x > 0 else 0.0)

            self.data.fillna({"click_times": 0, "buy_times": 0, "rating": 3.0}, inplace=True)
            print("Data loaded!")
            print(self.data)

    def preprocess_data(self, reload=False):
        """Tiền xử lý dữ liệu thành TensorFlow dataset."""
        if self.dataset is None or reload:
            print("Preprocessing dataset...")

            # Chuyển đổi categorical thành số
            self.data["gender"] = self.data["gender"].astype("category").cat.codes
            self.data["age_range"] = self.data["age_range"].astype("category").cat.codes
            self.data["type"] = self.data["type"].astype("category").cat.codes
            self.data["price_range"] = self.data["price_range"].astype("category").cat.codes
            self.data["location"] = self.data["location"].astype("category").cat.codes

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
        """Trả về dữ liệu dưới dạng pandas DataFrame."""
        return self.data

    def get_dataset(self):
        """Trả về dữ liệu dưới dạng TensorFlow dataset."""
        return self.dataset
    
    def get_product_ids(self):
        """Return product ids"""
        return self.data["product_id"].unique().tolist()

