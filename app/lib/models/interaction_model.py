from dataclasses import dataclass
from typing import Optional

@dataclass
class Interaction:
    id: str
    user_id: str
    item_id: str
    rating: int
    click_times: int
    buy_times: int

    @staticmethod
    def from_dict(interaction_id, data):
        return Interaction(
            id=interaction_id,
            user_id=data["userID"],
            item_id=data["itemID"],
            rating=data["rating"],
            click_times=data["clickTimes"],
            buy_times=data["buyTimes"],
        )