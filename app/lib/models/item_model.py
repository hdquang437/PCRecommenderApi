from dataclasses import dataclass
from typing import Optional
from datetime import datetime

@dataclass
class Item:
    id: str
    item_type: str
    price: int
    add_date: datetime = datetime.now()

    @staticmethod
    def from_dict(item_id, data):
        return Item(
            id=item_id,
            name=data["itemType"],
            email=data["price"],
            created_at=datetime.fromisoformat(data["addDate"])
        )