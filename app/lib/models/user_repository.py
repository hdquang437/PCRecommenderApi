import asyncio
from typing import Optional
from firebase_admin import firestore
from .user_model import User  # Import model đã tạo

class UserRepository:
    def __init__(self):
        self.db = firestore.client()
        self.collection = self.db.collection("Users")

    async def get_user(self, user_id: str) -> Optional[User]:
        """Lấy thông tin người dùng từ Firestore bất đồng bộ"""
        doc_ref = self.collection.document(user_id)
        doc = await doc_ref.get()
        if doc.exists:
            return User.from_dict(doc.to_dict())
        return None

    async def get_all_users(self):
        """Lấy danh sách tất cả người dùng bất đồng bộ"""
        users = []
        
        # Chuyển đổi thành async task với asyncio.to_thread để thực hiện trong luồng khác
        docs = await asyncio.to_thread(self.collection.stream)

        for doc in docs:
            users.append(User.from_dict(doc.to_dict()))
        
        return users
    
    
    def listen(self, callback):
        return self.collection.on_snapshot(callback)
