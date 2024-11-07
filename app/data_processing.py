import pandas as pd

def load_cache_data():
    users = pd.read_csv("train_data/cache_user.csv")
    items = pd.read_csv("train_data/cache_item.csv")
    interactions = pd.read_csv("train_data/cache_interaction.csv")
    deleted_items = pd.read_csv("train_data/cache_delete_item.csv")
    return users, items, interactions, deleted_items

def update_main_data():
    # Đọc dữ liệu từ cache
    users, items, interactions, deleted_items = load_cache_data()

    # Cập nhật vào các tệp chính
    main_users = pd.read_csv("train_data/user_data.csv")
    main_items = pd.read_csv("train_data/item_data.csv")
    main_interactions = pd.read_csv("train_data/interaction_data.csv")

    main_users = pd.concat([main_users, users]).drop_duplicates(subset=['userid'], keep="last")
    main_items = pd.concat([main_items, items]).drop_duplicates(subset=['itemid'], keep="last")
    main_interactions = pd.concat([main_interactions, interactions]).drop_duplicates(subset=['userid', 'itemid'], keep="last")

    # Cập nhật trạng thái xóa
    for item_id in deleted_items["itemid"]:
        main_items.loc[main_items["itemid"] == item_id, "isDelete"] = True

    # Lưu lại dữ liệu
    main_users.to_csv("train_data/user_data.csv", index=False)
    main_items.to_csv("train_data/item_data.csv", index=False)
    main_interactions.to_csv("train_data/interaction_data.csv", index=False)

    # Xóa dữ liệu trong cache
    pd.DataFrame(columns=users.columns).to_csv("train_data/cache_user.csv", index=False)
    pd.DataFrame(columns=items.columns).to_csv("train_data/cache_item.csv", index=False)
    pd.DataFrame(columns=interactions.columns).to_csv("train_data/cache_interaction.csv", index=False)
    pd.DataFrame(columns=deleted_items.columns).to_csv("train_data/cache_delete_item.csv", index=False)
