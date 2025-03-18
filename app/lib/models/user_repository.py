from typing import Optional
from firebase_admin import firestore
from user_model import User  # Import model đã tạo

class UserRepository:
    def __init__(self):
        self.db = firestore.client()
        self.collection = self.db.collection("users")

    def get_user(self, user_id: str) -> Optional[User]:
        """Lấy thông tin người dùng từ Firestore"""
        doc = self.collection.document(user_id).get()
        if doc.exists:
            return User.from_dict(doc.to_dict())
        return None

    def get_all_users(self):
        """Lấy danh sách tất cả người dùng"""
        return [User.from_dict(doc.to_dict()) for doc in self.collection.stream()]
