import datetime
from pymongo import MongoClient
from bson.objectid import ObjectId
import bcrypt
from app.models.user_model import User

class UserRepository:
    def __init__(self, uri: str, db_name: str):
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.users = self.db["users"]
    
    def login_user(self, email: str, password: str):
        """Authenticates a user and returns their UID if successful."""
        user = self.users.find_one({"email": email})
        if user and bcrypt.checkpw(password.encode('utf-8'), user["password"].encode('utf-8')):
            return str(user["_id"])
        return None
    
    def create_user(self, email: str, name: str, password: str):
        """Creates a new user and returns their UID."""
        user = User(name=name, email=email, password=password)
        user_data = user.to_dict()
        user_data["date_of_join"] = datetime.datetime.utcnow().isoformat()  
        result = self.users.insert_one(user_data)
        return str(result.inserted_id)
    
    def get_user(self, uid: str):
        """Fetches a user by UID."""
        user = self.users.find_one({"_id": ObjectId(uid)})
        if user:
            user["_id"] = str(user["_id"])  
            return user
        return None
    
    def update_user_data(self, uid: str, update_data: dict):
        """Updates user data based on provided fields."""
        self.users.update_one({"_id": ObjectId(uid)}, {"$set": update_data})
        return {"message": "User updated successfully"}
