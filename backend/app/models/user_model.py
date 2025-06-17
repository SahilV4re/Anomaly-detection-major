import bcrypt

class User:
    def __init__(self, name: str, email: str, password: str, settings: dict = None):
        self.name = name
        self.email = email
        self.password = self.hash_password(password)
        self.settings = settings if settings else {}

    def hash_password(self, password: str) -> bytes:
        """Hashes the password using bcrypt."""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt)

    def verify_password(self, password: str) -> bool:
        """Verifies the provided password against the stored hashed password."""
        return bcrypt.checkpw(password.encode('utf-8'), self.password)

    def to_dict(self):
        """Converts the user object to a dictionary for storage in MongoDB."""
        return {
            "name": self.name,
            "email": self.email,
            "password": self.password.decode('utf-8'),  # Convert bytes to string for MongoDB
            "settings": self.settings
        }

    @staticmethod
    def from_dict(data):
        """Creates a User object from a dictionary."""
        user = User(
            name=data.get("name", ""),
            email=data.get("email", ""),
            password=data.get("password", ""),
            settings=data.get("settings", {})
        )
        user.password = data.get("password", "").encode('utf-8')  # Store as bytes
        return user
