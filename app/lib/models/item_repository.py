import asyncio
from typing import Optional
from firebase_admin import firestore
from .item_model import Item  # Import model đã tạo

class ItemRepository:
    def __init__(self):
        self.db = firestore.client()
        self.collection = self.db.collection("Items")

    async def get_item(self, item_id: str) -> Optional[Item]:
        """Lấy thông tin item từ Firestore bất đồng bộ"""
        doc_ref = self.collection.document(item_id)
        doc = await doc_ref.get()
        if doc.exists:
            return Item.from_dict(doc.id, doc.to_dict())
        return None

    async def get_all_items(self):
        """Lấy danh sách tất cả items bất đồng bộ"""
        items = []
        
        # Chuyển đổi thành async task với asyncio.to_thread để thực hiện trong luồng khác
        docs = await asyncio.to_thread(self.collection.stream)

        for doc in docs:
            items.append(Item.from_dict(doc.id, doc.to_dict()))
        
        return items
    
    def listen(self, callback):
        return self.collection.on_snapshot(callback)
