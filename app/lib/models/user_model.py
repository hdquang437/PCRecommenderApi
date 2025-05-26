import json
from dataclasses import dataclass
from typing import Optional
from datetime import datetime

@dataclass
class User:
    id: str
    gender: str
    date_of_birth: datetime = datetime.now()

    @staticmethod
    def from_dict(data):
        return User(
            id=data["userID"],
            gender=data["gender"],
            date_of_birth=datetime.fromtimestamp(data["dateOfBirth"].timestamp())
        )