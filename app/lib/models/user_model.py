import json
from dataclasses import dataclass
from typing import Optional
from datetime import datetime

@dataclass
class User:
    id: str
    gender: str
    shop_name: str
    location: str
    date_of_birth: datetime = datetime.now()

    @staticmethod
    def from_dict(data):
        json_data = json.loads(data.get("shopInfo", {}))
        
        if not json_data:
            location = ""
            shop_name = ""
        else:
            location = json_data.get("location", "")
            shop_name = json_data.get("shopName", "")

        return User(
            id=data["userID"],
            gender=data["gender"],
            location=location,
            shop_name=shop_name,
            date_of_birth=datetime.fromtimestamp(data["dateOfBirth"].timestamp())
        )