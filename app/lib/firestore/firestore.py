import firebase_admin
from firebase_admin import credentials, firestore
import os
import json

# Load Firebase credentials

firebase_cred_json = os.getenv("FIREBASE_KEY_JSON")
cred_dict = json.loads(firebase_cred_json)

# cred = credentials.Certificate("./firebase_key.json")
cred = credentials.Certificate(cred_dict)
firebase_admin.initialize_app(cred)

# Khởi tạo Firestore
db = firestore.client()
