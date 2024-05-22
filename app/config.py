import os


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY") or "your-secret-key"
    SQLALCHEMY_DATABASE_URI = os.environ.get("DATABASE_URL") or "sqlite:///app.db"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    LOGGING_LEVEL = os.environ.get("LOGGING_LEVEL", "INFO")
    LOGGING_FORMAT = os.environ.get(
        "LOGGING_FORMAT", "[%(asctime)s] %(levelname)s in %(module)s: %(message)s"
    )
    LOGGING_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "logs")

    if not os.path.exists(LOGGING_DIR):
        os.makedirs(LOGGING_DIR)

    LOGGING_FILE = os.path.join(LOGGING_DIR, "app.log")

    if LOGGING_LEVEL == "DEBUG":
        LOGGING_FILE = os.path.join(LOGGING_DIR, "debug.log")

    LOGGING_HANDLER = {
        "handlers": {
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": LOGGING_FILE,
                "maxBytes": 1024 * 1024 * 5,  # 5 MB
                "backupCount": 5,
            }
        },
        "root": {"handlers": ["file"], "level": LOGGING_LEVEL},
    }

    if LOGGING_LEVEL == "DEBUG":
        LOGGING_HANDLER["handlers"]["file"]["level"] = "DEBUG"

    LOGGING_CONFIG = LOGGING_HANDLER
