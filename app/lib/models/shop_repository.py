import asyncio
from typing import Optional
from firebase_admin import firestore
from .shop_model import Shop  # Import model đã tạo

class ShopRepository:
    def __init__(self):
        self.db = firestore.client()
        self.collection = self.db.collection("Shops")

    async def get_shop(self, user_id: str) -> Optional[Shop]:
        """Lấy thông tin shop từ Firestore bất đồng bộ"""
        doc_ref = self.collection.document(user_id)
        doc = await doc_ref.get()
        if doc.exists:
            return Shop.from_dict(doc.id, doc.to_dict())
        return None

    async def get_all_shops(self):
        """Lấy danh sách tất cả shop bất đồng bộ"""
        shops = []
        
        # Chuyển đổi thành async task với asyncio.to_thread để thực hiện trong luồng khác
        docs = await asyncio.to_thread(self.collection.stream)

        for doc in docs:
            shops.append(Shop.from_dict(doc.id, doc.to_dict()))
        
        return shops
    
    
    def listen(self, callback):
        return self.collection.on_snapshot(callback)
