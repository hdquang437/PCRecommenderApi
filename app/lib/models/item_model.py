from dataclasses import dataclass
from typing import Optional
from datetime import datetime

@dataclass
class Item:
    id: str
    item_type: str
    seller_id: str
    price: int
    add_date: datetime = datetime.now()

    @staticmethod
    def from_dict(item_id, data):
        return Item(
            id=item_id,
            seller_id=data["sellerID"],
            item_type=data["itemType"],
            price=data["price"],
            add_date=datetime.fromtimestamp(data["addDate"].timestamp())
        )