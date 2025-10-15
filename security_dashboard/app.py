# IMPORT STATEMENTS
import os

from .core import SecuritySystem
from .web import app, socketio
from .web.routes import api_bp
from .utils import load_config, init_db

def main():
    # 1. Initialize DB and Config
    init_db()
    config = load_config()

    # 2. Validate camera configuration
    if not config.get('cameras') or len(config['cameras']) == 0:
        print("[CRITICAL ERROR] No cameras configured in config.json")
        print("Please add camera configuration. Example:")
        print("""
        "cameras": [
            {
                "camera_id": 0,
                "name": "Front Door",
                "width": 1280,
                "height": 720,
                "fps": 30,
                "enabled": true
            },
            {
                "camera_id": "rtsp://192.168.1.100/stream",
                "name": "Back Yard",
                "width": 1920,
                "height": 1080,
                "fps": 25,
                "enabled": true
            }
        ]
        """)
        return

    # 3. Initialize application
    security_system = SecuritySystem(config, socketio, app)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    app.config['PROJECT_ROOT'] = project_root
    app.security_system = security_system

    # 4. Register the blueprint containing all HTTP routes
    app.register_blueprint(api_bp)

    # 5. Start main processing
    if not security_system.start():
        print("[ERROR] Failed to start security system")
        return

    # 6. Start web server
    try:
        print("[STARTUP] Starting Flask-SocketIO server with multi-camera support...")
        debug_mode = os.environ.get('FLASK_DEBUG', 'false').lower() in ['true', '1', 't']
        socketio.run(app, host='0.0.0.0', port=5000, debug=debug_mode, use_reloader=debug_mode)
    except KeyboardInterrupt:
        print("Shutdown signal received. Cleaning up...")
    finally:
        security_system.stop()
        print("Cleanup complete. Exiting.")

if __name__ == "__main__":
    main()