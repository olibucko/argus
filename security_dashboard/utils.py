import json
import sqlite3
import threading


DB_FILE = "security_events.db"
CONFIG_FILE = "config.json"
config_lock = threading.Lock()

def load_config():
    with config_lock:
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)

def save_config(config_data):
    with config_lock:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config_data, f, indent=2)
    print("Configuration saved to config.json")

def init_db():
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

