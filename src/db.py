import os
from pathlib import Path
from typing import Any

from pymongo import MongoClient
from pymongo.collection import Collection

from src.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]
config = load_config(ROOT / "configs" / "config.yaml")

MONGO_URI = os.environ.get("MONGO_URI", config.get("database", {}).get("uri", "mongodb://127.0.0.1:27017"))
MONGO_NAME = os.environ.get("MONGO_NAME", config.get("database", {}).get("name", "mudra_recognition"))

client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
db = client[MONGO_NAME]

users_collection: Collection = db["users"]
tokens_collection: Collection = db["tokens"]

users_collection.create_index("username", unique=True)
users_collection.create_index("email", unique=True)
tokens_collection.create_index("token", unique=True)
