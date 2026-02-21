from flask import Flask
from flask_socketio import SocketIO
import os

# This navigates two directories up from this file's location (web/ -> security_dashboard/ -> root)
static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'static'))

app = Flask(__name__, template_folder='../templates', static_folder=static_dir)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading', logger=False, engineio_logger=False)

# Import handlers to register Socket.IO event handlers
# Must be imported AFTER socketio is created
from . import handlers