# IMPORT STATEMENTS
import os
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        logger.critical("No cameras configured in config.json")
        logger.critical("Please add camera configuration. See GitHub for example configuration.")
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
        logger.error("Failed to start security system")
        return

    # 6. Start web server
    try:
        logger.info("Starting Flask-SocketIO server with multi-camera support...")
        debug_mode = os.environ.get('FLASK_DEBUG', 'false').lower() in ['true', '1', 't']
        socketio.run(app, host='0.0.0.0', port=5000, debug=debug_mode, use_reloader=debug_mode)
    except KeyboardInterrupt:
        logger.info("Shutdown signal received. Cleaning up...")
    finally:
        security_system.stop()
        logger.info("Cleanup complete. Exiting.")

if __name__ == "__main__":
    main()