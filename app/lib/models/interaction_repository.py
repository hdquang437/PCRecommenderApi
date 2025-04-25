import asyncio
from typing import Optional
from firebase_admin import firestore
from .interaction_model import Interaction  # Import model đã tạo

class InteractionRepository:
    def __init__(self):
        self.db = firestore.client()
        self.collection = self.db.collection("Interactions")

    async def get_interaction(self, item_id: str) -> Optional[Interaction]:
        """Lấy thông tin interaction từ Firestore bất đồng bộ"""
        doc_ref = self.collection.document(item_id)
        doc = await doc_ref.get()
        if doc.exists:
            return Interaction.from_dict(doc.id, doc.to_dict())
        return None

    async def get_all_interactions(self):
        """Lấy danh sách tất cả interactions bất đồng bộ"""
        interactions = []
        
        # Chuyển đổi thành async task với asyncio.to_thread để thực hiện trong luồng khác
        docs = await asyncio.to_thread(self.collection.stream)

        for doc in docs:
            interactions.append(Interaction.from_dict(doc.id, doc.to_dict()))
        
        return interactions

    def listen(self, callback):
        return self.collection.on_snapshot(callback)