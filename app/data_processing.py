import pandas as pd
from .paths import USER_DATA_PATH, ITEM_DATA_PATH, INTERACTION_DATA_PATH, CACHE_DELETED_ITEM_PATH, CACHE_USER_PATH, CACHE_ITEM_PATH, CACHE_INTERACTION_PATH, CACHE_FIT_USER_ID, CACHE_FIT_ITEM_ID

class SingletonMeta(type):
    """
    This is a thread-safe implementation of Singleton.
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]
    
class DataStorage(metaclass=SingletonMeta):
    user_data = {}
    item_data = {}
    interaction_data = {}
    cache_user_data = {}
    cache_item_data = {}
    cache_interaction_data = {}
    cache_fit_item_ids = {}
    cache_fit_user_ids = {}

    def __init__(self):
        user_data = {}

def load_cache_data():
    users = pd.read_csv(CACHE_USER_PATH)
    items = pd.read_csv(CACHE_ITEM_PATH)
    interactions = pd.read_csv(CACHE_INTERACTION_PATH)
    deleted_items = pd.read_csv(CACHE_DELETED_ITEM_PATH)
    return users, items, interactions, deleted_items

def load_main_data():
    user_data = pd.read_csv(USER_DATA_PATH)
    item_data = pd.read_csv(ITEM_DATA_PATH)
    interaction_data = pd.read_csv(INTERACTION_DATA_PATH)
    return user_data, item_data, interaction_data

def update_main_data():
    # Đọc dữ liệu từ cache
    users, items, interactions, deleted_items = load_cache_data()

    # Cập nhật vào các tệp chính
    main_users, main_items, main_interactions = load_main_data()

    main_users = pd.concat([main_users, users]).drop_duplicates(subset=['userid'], keep="last")
    main_items = pd.concat([main_items, items]).drop_duplicates(subset=['itemid'], keep="last")
    main_interactions = pd.concat([main_interactions, interactions]).drop_duplicates(subset=['userid', 'itemid'], keep="last")

    # Cập nhật trạng thái xóa
    for item_id in deleted_items["itemid"]:
        main_items.loc[main_items["itemid"] == item_id, "is_delete"] = 1

    # Lưu lại dữ liệu
    main_users.to_csv(USER_DATA_PATH, index=False)
    main_items.to_csv(ITEM_DATA_PATH, index=False)
    main_interactions.to_csv(INTERACTION_DATA_PATH, index=False)

    # Xóa dữ liệu trong cache
    pd.DataFrame(columns=users.columns).to_csv(CACHE_USER_PATH, index=False)
    pd.DataFrame(columns=items.columns).to_csv(CACHE_ITEM_PATH, index=False)
    pd.DataFrame(columns=interactions.columns).to_csv(CACHE_INTERACTION_PATH, index=False)
    pd.DataFrame(columns=deleted_items.columns).to_csv(CACHE_DELETED_ITEM_PATH, index=False)

# Hàm lưu trữ mapping giữa id sau khi fit-transform và id thật
def cache_fit_ids(user_data, item_data):
    cache_fit_user_ids = pd.DataFrame({
        "userid": []       
    })
    cache_fit_user_ids["userid"] = user_data[:, 0]
    cache_fit_user_ids.to_csv(CACHE_FIT_USER_ID, index=True, index_label="index")
    cache_fit_item_ids = pd.DataFrame({
        "itemid": []       
    })
    cache_fit_item_ids["itemid"] = item_data[:, 0]
    cache_fit_item_ids.to_csv(CACHE_FIT_ITEM_ID, index=True, index_label="index")

# Hàm đọc mapping giữa id sau khi fit-tranform và id thật
def read_cached_fit_ids():
    fit_user_ids = pd.read_csv(CACHE_FIT_USER_ID)
    fit_item_ids = pd.read_csv(CACHE_FIT_ITEM_ID)
    return fit_user_ids, fit_item_ids