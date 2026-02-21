import json
import os
import sqlite3
import threading
import logging

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables from .env file (if present)
load_dotenv()

DB_FILE = "security_events.db"
CONFIG_FILE = "config.json"
config_lock = threading.Lock()

def _overlay_secrets(config: dict) -> dict:
    """
    Overlay secret values from environment variables onto the config dict.
    Environment variables take precedence over config.json values.
    """
    am = config.get("alert_manager", {})

    # Telegram secrets
    if os.getenv("TELEGRAM_BOT_TOKEN"):
        am["telegram_bot_token"] = os.getenv("TELEGRAM_BOT_TOKEN")
    if os.getenv("TELEGRAM_CHAT_IDS"):
        am["telegram_chat_ids"] = [cid.strip() for cid in os.getenv("TELEGRAM_CHAT_IDS").split(",")]

    # Email secrets
    if os.getenv("EMAIL_PASSWORD"):
        am["email_password"] = os.getenv("EMAIL_PASSWORD")
    if os.getenv("SENDER_EMAIL"):
        am["sender_email"] = os.getenv("SENDER_EMAIL")
    if os.getenv("EMAIL_RECIPIENTS"):
        am["recipient_list"] = [r.strip() for r in os.getenv("EMAIL_RECIPIENTS").split(",")]

    config["alert_manager"] = am

    # Camera RTSP URLs (CAMERA_1_URL, CAMERA_2_URL, ...)
    for i, camera in enumerate(config.get("cameras", []), start=1):
        env_url = os.getenv(f"CAMERA_{i}_URL")
        if env_url:
            camera["camera_id"] = env_url

    return config

def load_config() -> dict:
    """
    Load the main configuration from config.json in a thread-safe manner.
    Secret values are overlaid from environment variables (.env file).

    Returns:
        dict: The application configuration with secrets applied.
    """
    with config_lock:
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
        return _overlay_secrets(config)

# Fields that are managed by .env and must not be written back to config.json.
_SECRET_FIELDS = {"telegram_bot_token", "telegram_chat_ids", "email_password", "sender_email", "recipient_list"}

def _strip_secrets(config_data: dict) -> dict:
    """Return a deep copy of config with secret fields reset to safe defaults."""
    import copy
    clean = copy.deepcopy(config_data)
    am = clean.get("alert_manager", {})
    for field in _SECRET_FIELDS:
        if field in am:
            am[field] = [] if isinstance(am[field], list) else ""
    for camera in clean.get("cameras", []):
        if os.getenv(f"CAMERA_{clean['cameras'].index(camera) + 1}_URL"):
            camera["camera_id"] = ""
    return clean

def save_config(config_data: dict) -> None:
    """
    Save the configuration data to config.json in a thread-safe manner.
    Secret values sourced from .env are stripped before writing.

    Args:
        config_data (dict): The configuration data to save.
    """
    clean_data = _strip_secrets(config_data)
    with config_lock:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(clean_data, f, indent=2)
    logger.info("Configuration saved to config.json")

def init_db() -> None:
    """
    Initialize the SQLite database and create the 'events' table if it doesn't exist.
    """
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            viewport_name TEXT NOT NULL,
            confidence INTEGER NOT NULL,
            screenshot_path TEXT,
            video_path TEXT
        )
    ''')
    conn.commit()
    conn.close()
