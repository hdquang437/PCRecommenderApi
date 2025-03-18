import json
from dataclasses import dataclass
from typing import Optional
from datetime import datetime

@dataclass
class User:
    id: str
    gender: str
    shop_name: str
    date_of_birth: datetime = datetime.now()

    @staticmethod
    def from_dict(data):
        json_data = json.loads(data.get("shopInfo", "{}"))

        return User(
            id=data["userID"],
            gender=data["gender"],
            location=json_data.get("location"),
            date_of_birth=datetime.fromisoformat(data["dateOfBirth"])
        )