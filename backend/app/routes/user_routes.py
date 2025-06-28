from flask import Blueprint, request, jsonify
from app.repositories.user_repo import UserRepository

# MongoDB Credentials
MONGO_URI = "mongodb+srv://sahil:sahilsv@mongocluster.zceoy2s.mongodb.net/?retryWrites=true&w=majority&appName=mongocluster"
DB_NAME = "user_anomaly"

user_bp = Blueprint("user_routes", __name__)
repo = UserRepository(MONGO_URI, DB_NAME)

@user_bp.route("/login", methods=["POST"])
def login():
    try:
        data = request.json
        email = data.get("email")
        password = data.get("password")
        
        if not email or not password:
            return jsonify({"error": "Email and password are required"}), 400
        
        uid = repo.login_user(email, password)
        if uid:
            return jsonify({"uid": uid}), 200
        return jsonify({"error": "Invalid credentials"}), 401
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@user_bp.route("/create_user", methods=["POST"])
def create_user():
    try:
        data = request.json
        email = data.get("email")
        name = data.get("name")
        password = data.get("password")
        
        if not email or not name or not password:
            return jsonify({"error": "Email, name, and password are required"}), 400
        
        uid = repo.create_user(email, name, password)
        return jsonify({"uid": uid}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@user_bp.route("/get_user/<uid>", methods=["GET"])
def get_user(uid):
    try:
        user = repo.get_user(uid)
        if user:
            return jsonify({"data": user}), 200
        return jsonify({"error": "User not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@user_bp.route("/update_user", methods=["POST"])
def update_user():
    try:
        data = request.json
        uid = data.get("uid")
        update_data = data.get("update_data")  # assuming you pass only the fields to update here

        if not uid or not update_data:
            return jsonify({"error": "UID and update_data are required"}), 400
        
        repo.update_user_data(uid, update_data)
        return jsonify({"message": "User updated successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
