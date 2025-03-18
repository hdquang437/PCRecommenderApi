from typing import Optional
from firebase_admin import firestore
from item_model import Item  # Import model đã tạo

class ItemRepository:
    def __init__(self):
        self.db = firestore.client()
        self.collection = self.db.collection("users")

    def get_item(self, item_id: str) -> Optional[Item]:
        doc = self.collection.document(item_id).get()
        if doc.exists:
            return Item.from_dict(doc.id, doc.to_dict())
        return None

    def get_all_items(self):
        return [Item.from_dict(doc.id, doc.to_dict()) for doc in self.collection.stream()]
