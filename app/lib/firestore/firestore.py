import firebase_admin
from firebase_admin import credentials, firestore

# Load Firebase credentials
cred = credentials.Certificate("./firebase_key.json")
firebase_admin.initialize_app(cred)

# Khởi tạo Firestore
db = firestore.client()
