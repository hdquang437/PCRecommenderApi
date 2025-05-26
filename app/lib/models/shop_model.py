import json
from dataclasses import dataclass
from typing import Optional
from datetime import datetime

@dataclass
class Shop:
    id: str
    shop_name: str
    location: str

    @staticmethod
    def from_dict(shop_id, data):
        return Shop(
            id=shop_id,
            location=data["location"],
            shop_name=data["name"]
        )