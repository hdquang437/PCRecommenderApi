from typing import Optional
from firebase_admin import firestore
from interaction_model import Interaction  # Import model đã tạo

class InteractionRepository:
    def __init__(self):
        self.db = firestore.client()
        self.collection = self.db.collection("interactions")

    def get_interaction(self, item_id: str) -> Optional[Interaction]:
        doc = self.collection.document(item_id).get()
        if doc.exists:
            return Interaction.from_dict(doc.id, doc.to_dict())
        return None

    def get_all_items(self):
        return [Interaction.from_dict(doc.id, doc.to_dict()) for doc in self.collection.stream()]
