from flask import Flask
from flask_cors import CORS
from app.routes.user_routes import user_bp

app = Flask(__name__)
app.register_blueprint(user_bp, url_prefix="/user")

cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
cors = CORS(app, resources={r"/*": {"origins": "*"}})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)