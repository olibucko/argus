#### IMPORTS ###
import time
import os
import logging
import asyncio
import subprocess
from datetime import datetime, time as dt_time
from dataclasses import dataclass, asdict
from collections import defaultdict
import threading
from queue import Queue, Empty, Full
from typing import Tuple, List, Optional, Dict

import cv2
import numpy as np
import yagmail
from flask_socketio import SocketIO
import sqlite3

logger = logging.getLogger(__name__)

try:
    from telegram import Bot
    from telegram.error import TelegramError
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    logger.warning("python-telegram-bot not installed. Telegram alerts will be disabled.")

from .yolo_process import YOLOProcessManager
from .motion_detector import MotionDetector, MotionDetectorConfig
from .camera_manager import CameraManager, create_camera_configs
from .frame_broker import FrameBroker
from .memory_manager import MemoryBoundedBuffer, GlobalMemoryManager, EvictionPolicy
from .frame_pool import SharedFramePool
from .metrics import get_global_metrics, record_metric


#### CONSTANTS ###
DB_FILE = "security_events.db"
MAX_CONCURRENT_RECORDINGS = 4

### HELPER FUNCTIONS ###
def calculate_optimal_grid(num_cameras: int) -> Tuple[int, int]:
    """Calculate optimal grid dimensions for the given number of cameras."""
    if num_cameras <= 0:
        return (1, 1)
    if num_cameras == 1:
        return (1, 1)
    if num_cameras <= 4:
        return (2, 2)
    if num_cameras <= 6:
        return (2, 3)
    if num_cameras <= 9:
        return (3, 3)
    if num_cameras <= 12:
        return (3, 4)
    if num_cameras <= 16:
        return (4, 4)
    if num_cameras <= 20:
        return (4, 5)
    # For more cameras, keep aspect ratio reasonable
    cols = int(np.ceil(np.sqrt(num_cameras * 1.6)))  # Slightly wider than square
    rows = int(np.ceil(num_cameras / cols))
    return (rows, cols)

def create_viewport_configs_from_cameras(cameras: List[dict], viewport_defaults: dict) -> Dict[str, dict]:
    """Create viewport configurations from camera list with dynamic grid positioning.
    Each camera outputs one full resolution stream (no virtual viewports)."""
    enabled_cameras = [cam for cam in cameras if cam.get('enabled', True)]

    if not enabled_cameras:
        return {}

    # Calculate grid dimensions - one viewport per camera
    total_viewports = len(enabled_cameras)
    grid_size = calculate_optimal_grid(total_viewports)
    rows, cols = grid_size

    viewport_configs = {}

    for viewport_index, camera in enumerate(enabled_cameras):
        # Calculate grid position for this camera
        row = viewport_index // cols
        col = viewport_index % cols
        viewport_id = f"{row},{col}"

        # Merge camera-specific settings with defaults
        config = viewport_defaults.copy()
        # Prefer custom 'name' field, fallback to 'camera_name' for display
        camera_name_value = camera.get('name', camera.get('camera_name'))
        config.update({
            'camera_name': camera.get('camera_name'),  # Keep original camera identifier
            'name': camera_name_value,  # Frontend expects 'name' field
            'sensitivity': camera.get('sensitivity', config['sensitivity']),
            'min_confidence': camera.get('min_confidence', config['min_confidence']),
            'yolo_interval': camera.get('yolo_interval', config['yolo_interval']),
            'motion_aggressiveness': camera.get('motion_aggressiveness', config['motion_aggressiveness']),
            'frame_skip': camera.get('frame_skip', config['frame_skip']),
            'min_object_size': camera.get('min_object_size', config['min_object_size']),
            'motion_threshold': camera.get('motion_threshold', config['motion_threshold']),
            'scale_factor': camera.get('scale_factor', config['scale_factor'])
        })

        viewport_configs[viewport_id] = config

    return viewport_configs

### DATACLASSES ###
@dataclass
class ViewportConfig:
    camera_name: str = "Unnamed"
    name: str = "Unnamed"  # Frontend display name
    sensitivity: float = 0.5
    min_confidence: float = 0.5
    yolo_interval: int = 5
    motion_aggressiveness: float = 25.0
    motion_threshold: int = 30
    min_object_size: Tuple[int, int] = (30, 30)
    frame_skip: int = 1
    scale_factor: float = 0.5

@dataclass
class AlertConfig:
    curfew_start: dt_time
    curfew_end: dt_time
    cooldown_period: float
    batch_window: float
    recipient_list: List[str]
    timezone: str = "AEST"
    sender_email: str = None
    sender_name: str = None
    email_password: str = None
    # Telegram configuration
    telegram_bot_token: Optional[str] = None
    telegram_chat_ids: Optional[List[str]] = None
    telegram_enabled: bool = False
    email_as_fallback: bool = True

### CORE CLASSES ###
class VideoClipRecorder:
    def __init__(self, config: dict, frame_width: int, frame_height: int, fps: int, shared_lock: threading.RLock):
        self.video_dir = "event_videos"
        os.makedirs(self.video_dir, exist_ok=True)
        self.pre_event_seconds = config['pre_event_seconds']
        self.post_event_seconds = config['post_event_seconds']
        self.max_clip_duration = config['max_clip_duration_seconds']
        self.codec = cv2.VideoWriter_fourcc(*config['codec'])
        self.frame_width, self.frame_height, self.fps = frame_width, frame_height, fps

        # Test codec compatibility on startup
        test_path = "test_codec.mp4"
        test_writer = cv2.VideoWriter(test_path, self.codec, self.fps, (self.frame_width, self.frame_height))
        if not test_writer.isOpened():
            logger.error(f"Codec '{config['codec']}' is not compatible with your system!")
            logger.error("Video recording will fail. Please change codec in config.json to 'mp4v' or 'XVID'")
            print(f"[ERROR] Video codec '{config['codec']}' not supported - recordings will fail!")
        else:
            logger.info(f"Video codec '{config['codec']}' validated successfully (for web viewing)")
            logger.info("Telegram clips will be auto-converted to H.264 for compatibility")
        test_writer.release()
        if os.path.exists(test_path):
            os.remove(test_path)

        self.recording_state: Dict[Tuple[int, int], Dict] = {}
        self.lock = shared_lock

    def update_dimensions(self, frame_width: int, frame_height: int):
        """Update the video recorder dimensions based on actual frame sizes."""
        self.frame_width = frame_width
        self.frame_height = frame_height
        logger.info(f"VideoClipRecorder dimensions updated to {frame_width}x{frame_height}")

    def is_recording(self, viewport_id: Tuple[int, int]) -> bool:
        with self.lock:
            state = self.recording_state.get(viewport_id)
            return state and not state['done'].is_set()

    def update_activity(self, viewport_id: Tuple[int, int]):
        with self.lock:
            state = self.recording_state.get(viewport_id)
            if state:
                state['last_activity_time'] = time.time()

    def handle_alert_event(self, event: Dict, recording_buffer) -> Optional[str]:
        """Checks recording state and starts or updates a recording based on the event."""
        viewport_id = event['viewport_id']
        if self.is_recording(viewport_id):
            self.update_activity(viewport_id)
            return self.recording_state[viewport_id]['filepath']
        else:
            timestamp = datetime.fromtimestamp(event['timestamp'])
            viewport_name = event['viewport_name']
            return self.start_recording(viewport_id, timestamp, viewport_name, recording_buffer)

    def start_recording(self, viewport_id: Tuple[int, int], timestamp: datetime, viewport_name: str, recording_buffer) -> Optional[str]:
        with self.lock:
            if self.is_recording(viewport_id):
                self.update_activity(viewport_id)
                return self.recording_state[viewport_id]['filepath']

            if len(self.recording_state) >= MAX_CONCURRENT_RECORDINGS:
                return None

            done_event = threading.Event()
            filename = f"{timestamp.strftime('%Y%m%d_%H%M%S')}_{viewport_name.replace(' ', '_')}.mp4"
            filepath = os.path.join(self.video_dir, filename)

            q_size = int(self.max_clip_duration * self.fps * 1.2)
            frame_queue = Queue(maxsize=q_size)

            current_time = time.time()
            self.recording_state[viewport_id] = {
                'filepath': filepath,
                'done': done_event,
                'start_time': current_time,
                'last_activity_time': current_time,
                'frame_queue': frame_queue
            }

            # Get pre-event frames from recording buffer
            pre_event_frames = recording_buffer.get_all_frames()

            thread = threading.Thread(target=self._write_video_thread, args=(viewport_id, filepath, pre_event_frames))
            thread.daemon = True
            thread.start()

            return filepath

    def _write_video_thread(self, viewport_id: Tuple[int, int], filepath: str, pre_event_frames: List[np.ndarray]):
        with self.lock:
            state = self.recording_state.get(viewport_id)
        if not state:
            print(f"Error: Could not find recording state for {viewport_id}")
            return

        out = cv2.VideoWriter(filepath, self.codec, self.fps, (self.frame_width, self.frame_height))

        if not out.isOpened():
            logger.error(f"Failed to open video writer for {filepath}")
            logger.error(f"Codec fourcc: {self.codec}, FPS: {self.fps}, Resolution: {self.frame_width}x{self.frame_height}")
            print(f"[ERROR] Failed to open video writer for {filepath} - check codec compatibility")
            return

        # Write pre-event frames (already resized in recording buffer)
        for frame in pre_event_frames:
            out.write(frame)

        while True:
            now = time.time()
            with self.lock:
                last_activity = state['last_activity_time']
                start_time = state['start_time']

            if now - last_activity > self.post_event_seconds:
                print(f"Finishing recording for {filepath} due to inactivity.")
                break
            if now - start_time > self.max_clip_duration:
                print(f"Finishing recording for {filepath} due to max duration.")
                break

            try:
                frame = state['frame_queue'].get(timeout=1.0)
                # Frames are already resized when added to recording buffer
                out.write(frame)
            except Empty:
                continue

        # Write remaining frames in queue
        while not state['frame_queue'].empty():
            try:
                frame = state['frame_queue'].get_nowait()
                out.write(frame)
            except Empty:
                break

        out.release()
        print(f"Saved video clip to {filepath}")
        with self.lock:
            state['done'].set()
            self.recording_state.pop(viewport_id, None)

class AlertManager:
    def __init__(self, config: AlertConfig, socketio_instance: SocketIO, shared_lock: threading.RLock, video_recorder: 'VideoClipRecorder', flask_app=None):
        self.config = config
        self.socketio = socketio_instance
        self.flask_app = flask_app  # Store Flask app for application context
        self.alert_lock = shared_lock
        self.video_recorder = video_recorder
        self.last_alert_times: Dict[tuple, float] = defaultdict(float)
        self.pending_alerts: Dict[tuple, List[Dict]] = defaultdict(list)
        self.batch_timers: Dict[tuple, Optional[threading.Timer]] = {}
        os.makedirs("event_captures", exist_ok=True)

        self.detection_enabled = False

        # Telegram configuration validation and testing
        self.telegram_enabled = False
        self.telegram_bot = None
        self._validate_and_test_telegram_config()

        # Email configuration validation and testing
        self.email_enabled = False
        self._validate_and_test_email_config()

        # Database connection pooling - use queue for batch writes
        self.db_queue = Queue()
        self.db_worker_running = True
        self.db_worker_thread = threading.Thread(target=self._db_writer_worker, daemon=True)
        self.db_worker_thread.start()

    def _validate_and_test_telegram_config(self) -> None:
        """Validate Telegram bot configuration and test connection."""
        if not TELEGRAM_AVAILABLE:
            logger.warning("Telegram library not available - alerts disabled")
            return

        if not self.config.telegram_enabled:
            logger.info("Telegram alerts disabled in configuration")
            return

        if not self.config.telegram_bot_token:
            logger.warning("Telegram bot token not configured - alerts disabled")
            return

        if not self.config.telegram_chat_ids or len(self.config.telegram_chat_ids) == 0:
            logger.warning("No Telegram chat IDs configured - alerts disabled")
            return

        # Test bot connection
        logger.info("Testing Telegram bot configuration...")
        try:
            self.telegram_bot = Bot(token=self.config.telegram_bot_token)

            # Test connection by getting bot info (synchronous call)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                bot_info = loop.run_until_complete(self.telegram_bot.get_me())
                self.telegram_enabled = True
                logger.info(f"Telegram bot validated: @{bot_info.username} ({bot_info.first_name})")
                logger.info(f"Will send alerts to {len(self.config.telegram_chat_ids)} chat(s)")
            finally:
                loop.close()

        except Exception as e:
            logger.error(f"Failed to validate Telegram bot: {e}")
            logger.error("Telegram alerts will be disabled. Check your bot token.")
            self.telegram_bot = None

    def _validate_and_test_email_config(self) -> None:
        """Validate email configuration and test SMTP connection."""
        # Check if email credentials are provided
        if not self.config.sender_email or not self.config.email_password:
            logger.warning("Email credentials not configured - email alerts disabled")
            return

        if not self.config.recipient_list:
            logger.warning("No email recipients configured - email alerts disabled")
            return

        # Test SMTP connection
        logger.info("Testing email configuration...")
        try:
            yag = yagmail.SMTP(user=self.config.sender_email, password=self.config.email_password)
            yag.close()
            self.email_enabled = True
            logger.info(f"Email configuration validated successfully. Recipients: {self.config.recipient_list}")
        except Exception as e:
            logger.error(f"Failed to validate email configuration: {e}")
            logger.error("Email alerts will be disabled. Please check your credentials.")

    def is_curfew_hours(self) -> bool:
        current_time = datetime.now().time()
        if self.config.curfew_start <= self.config.curfew_end:
            return self.config.curfew_start <= current_time <= self.config.curfew_end
        else:
            return current_time >= self.config.curfew_start or current_time <= self.config.curfew_end

    def handle_alert_event(self, event: Dict):
        """Public entry point called by the event handler worker."""
        if not self.detection_enabled:
            return

        viewport_id = event['viewport_id']
        with self.alert_lock:
            
            # 1. First, check if a batch is already in progress. If so, just add to it.
            if viewport_id in self.batch_timers and self.batch_timers[viewport_id] and self.batch_timers[viewport_id].is_alive():
                self.pending_alerts[viewport_id].append(event)
                # Keep any active recording alive
                if self.video_recorder.is_recording(viewport_id):
                    self.video_recorder.update_activity(viewport_id)
                return

            # 2. If no batch is active, check the cooldown period.
            current_time = time.time()
            if current_time - self.last_alert_times[viewport_id] < self.config.cooldown_period:
                # During cooldown, we don't start a NEW alert, but we must keep the video recording if it's active.
                if self.video_recorder.is_recording(viewport_id):
                    self.video_recorder.update_activity(viewport_id)
                return

            # 3. If cooldown has passed, initiate a new alert batch.
            self._initiate_new_event_batch(event)

    def _initiate_new_event_batch(self, event: Dict):
        """Starts the batching window for a new event."""
        viewport_id = event['viewport_id']
        
        # Take a screenshot for the first detection in a batch
        timestamp = datetime.fromtimestamp(event['timestamp'])
        screenshot_path = self._take_screenshot(event['frame'], timestamp, event['viewport_name'])
        event['screenshot_path'] = screenshot_path

        # Add the first event to the pending list and start the batching timer
        self.pending_alerts[viewport_id].append(event)
        timer = threading.Timer(self.config.batch_window, self._process_and_send_alert, args=[viewport_id])
        timer.daemon = True
        timer.start()
        self.batch_timers[viewport_id] = timer
        
        # Use application context when emitting from a background thread
        if self.flask_app:
            with self.flask_app.app_context():
                self.socketio.emit('detection_in_progress', {'row': viewport_id[0], 'col': viewport_id[1]}, namespace='/')
        else:
            # Fallback for when app context is not available (e.g., in tests)
            self.socketio.emit('detection_in_progress', {'row': viewport_id[0], 'col': viewport_id[1]}, namespace='/')

    def _take_screenshot(self, frame, timestamp, viewport_name):
        """Take a screenshot and return the path, or None if failed."""
        try:
            snapshot_filename = f"{timestamp.strftime('%Y%m%d_%H%M%S')}_{viewport_name.replace(' ', '_')}.jpg"
            snapshot_path = os.path.join("event_captures", snapshot_filename)
            success = cv2.imwrite(snapshot_path, frame)
            if success:
                logger.info(f"Screenshot saved: {snapshot_path}")
                return snapshot_path
            else:
                logger.error(f"Failed to save screenshot: {snapshot_path}")
                return None
        except Exception as e:
            logger.error(f"Error taking screenshot for {viewport_name}: {e}")
            return None

    def _log_event_to_db(self, timestamp, viewport_name, confidence, screenshot_path, video_path):
        """Queue event for batch database writing."""
        self.db_queue.put((timestamp, viewport_name, confidence, screenshot_path, video_path))

    def _db_writer_worker(self):
        """Background worker that batches database writes for efficiency."""
        batch = []
        batch_timeout = 1.0  # Write batch every 1 second
        last_write_time = time.time()

        while self.db_worker_running:
            try:
                # Collect events for batching
                try:
                    event = self.db_queue.get(timeout=0.1)
                    batch.append(event)
                except Empty:
                    pass

                # Write batch if timeout reached or batch is full
                current_time = time.time()
                if batch and (len(batch) >= 10 or (current_time - last_write_time) >= batch_timeout):
                    try:
                        conn = sqlite3.connect(DB_FILE, check_same_thread=False)
                        cursor = conn.cursor()

                        # Batch insert all events
                        cursor.executemany(
                            "INSERT INTO events (timestamp, viewport_name, confidence, screenshot_path, video_path) VALUES (?, ?, ?, ?, ?)",
                            [(ts.isoformat(), vp, conf, ss, vid) for ts, vp, conf, ss, vid in batch]
                        )

                        conn.commit()
                        conn.close()

                        print(f"Logged {len(batch)} events to database in batch.")
                        batch = []
                        last_write_time = current_time

                    except Exception as e:
                        print(f"Database batch write error: {e}")
                        batch = []  # Clear failed batch

            except Exception as e:
                print(f"Error in database writer worker: {e}")
                time.sleep(1.0)

    def _process_and_send_alert(self, viewport_id: tuple):
        alerts_to_process = []
        
        with self.alert_lock:
            if not self.pending_alerts.get(viewport_id):
                return
            
            alerts_to_process = self.pending_alerts.pop(viewport_id, [])
            self.last_alert_times[viewport_id] = time.time()
            self.batch_timers.pop(viewport_id, None)

        if not alerts_to_process:
            return

        primary_alert = max(alerts_to_process, key=lambda x: x['confidence'])
        primary_timestamp = datetime.fromtimestamp(primary_alert['timestamp'])
        screenshot_path = primary_alert.get('screenshot_path')
        video_path = primary_alert.get('video_path')  # Full-length original video

        # Log to database and emit to frontend (uses full video for web viewing)
        self._log_event_to_db(
            timestamp=primary_timestamp,
            viewport_name=primary_alert['viewport_name'],
            confidence=primary_alert['confidence'],
            screenshot_path=screenshot_path,
            video_path=video_path  # Full video for event log
        )

        alert_to_emit = {
            'viewport_name': primary_alert['viewport_name'],
            'timestamp': primary_timestamp.isoformat(),
            'confidence': primary_alert['confidence'],
            'row': viewport_id[0],
            'col': viewport_id[1],
            'video_path': video_path  # Full video for frontend
        }

        if self.flask_app:
            with self.flask_app.app_context():
                self.socketio.emit('new_alert', alert_to_emit, namespace='/')
                print(f"[SOCKET.IO] Emitted new_alert: {alert_to_emit}")
        else:
            self.socketio.emit('new_alert', alert_to_emit, namespace='/')
            print(f"[SOCKET.IO] NO CONTEXT Emitted new_alert: {alert_to_emit}")

        # Alert sending logic during curfew hours: Telegram first, email as fallback
        is_curfew = self.is_curfew_hours()
        logger.info(f"Processing alert for {primary_alert['viewport_name']} - Curfew hours: {is_curfew}, Telegram enabled: {self.telegram_enabled}")

        if is_curfew:
            telegram_sent = False

            # Try Telegram first (primary alert mechanism)
            if self.telegram_enabled:
                # Wait for video recording to complete before extraction
                if video_path and self.video_recorder.is_recording(viewport_id):
                    logger.info("Waiting for video recording to complete before extraction...")
                    with self.video_recorder.lock:
                        state = self.video_recorder.recording_state.get(viewport_id)
                    if state:
                        # Wait up to 60 seconds for recording to finish
                        if state['done'].wait(timeout=60):
                            logger.info("Video recording completed, proceeding with extraction")
                        else:
                            logger.warning("Video recording timeout - proceeding anyway")

                # Extract video clip if available (limit to 40MB for Telegram)
                # NOTE: Clip is temporary and will be deleted after sending
                video_clip_path = None
                if video_path:
                    video_clip_path = self._extract_video_clip(video_path, max_size_mb=40)
                    if not video_clip_path:
                        logger.warning("Video extraction failed, sending screenshot only")

                telegram_message = self._prepare_telegram_message(alerts_to_process, video_path)
                telegram_sent = self._send_telegram_alert(telegram_message, screenshot_path, video_clip_path)

                if telegram_sent:
                    logger.info("Alert sent via Telegram (primary)")
                else:
                    logger.warning("Telegram alert failed")

            # Send email as fallback or if configured to send both
            should_send_email = False

            if self.email_enabled:
                if not self.telegram_enabled:
                    # Telegram not configured, use email
                    should_send_email = True
                    logger.info("Using email alerts (Telegram not configured)")
                elif not telegram_sent and self.config.email_as_fallback:
                    # Telegram failed and fallback is enabled
                    should_send_email = True
                    logger.info("Using email as fallback (Telegram failed)")

            if should_send_email:
                content, attachments = self._prepare_alert_content(
                    alerts_to_process,
                    permanent_path=screenshot_path,
                    video_path=video_path
                )
                self._send_email(
                    subject=f"Security Alert - {primary_alert['viewport_name']}",
                    content=content,
                    attachments=attachments
                )

            if not telegram_sent and not should_send_email:
                logger.warning("No alerts sent - both Telegram and email unavailable or disabled")

    def _prepare_alert_content(self, alerts: List[Dict], permanent_path: Optional[str], video_path: Optional[str] = None) -> Tuple[str, List[str]]:
        viewport_name = alerts[0]['viewport_name']
        timestamp = datetime.fromtimestamp(alerts[0]['timestamp'])
        num_detections = len(alerts)
        colors = {"bg_dark": "#09090b", "card_bg": "#18181b", "border": "#27272a", "heading_text": "#fafafa", "body_text": "#d4d4d8", "accent_red": "#f87171"}

        # Build attachment status message
        attachment_msg = ""
        if permanent_path:
            attachment_msg = "A screenshot of the event is attached to this email for your review."
        else:
            attachment_msg = "Screenshot capture failed - please check the system logs."

        if video_path:
            attachment_msg += " A video clip has been saved locally on the system."

        content = f"""
        <!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>Security Alert: {viewport_name}</title></head>
        <body style="margin: 0; padding: 0; width: 100% !important; background-color: {colors['bg_dark']};">
            <table align="center" border="0" cellpadding="0" cellspacing="0" width="100%" style="max-width: 600px; margin: 0 auto;">
                <tr><td align="center" style="padding: 20px 0;">
                    <table align="center" border="0" cellpadding="0" cellspacing="0" width="100%" style="background-color: {colors['card_bg']}; border-radius: 12px; border: 1px solid {colors['border']}; padding: 30px 40px;">
                        <tr><td align="center" style="padding-bottom: 20px; border-bottom: 1px solid {colors['border']};"><h1 style="margin: 0; font-family: Inter, Arial, sans-serif; color: {colors['heading_text']}; font-size: 24px; font-weight: bold; letter-spacing: 1px;">Argus</h1></td></tr>
                        <tr><td style="padding: 15px 0;"></td></tr>
                        <tr><td align="center"><h2 style="margin: 0; font-family: Inter, Arial, sans-serif; color: {colors['accent_red']}; font-size: 20px; font-weight: 600;">SECURITY ALERT: {viewport_name.upper()}</h2></td></tr>
                        <tr><td style="padding: 15px 0;"></td></tr>
                        <tr><td style="font-family: Inter, Arial, sans-serif; color: {colors['body_text']}; font-size: 16px; line-height: 1.6;">
                            An intrusion has been detected with the following details:
                            <br><br>&bull; <strong>Time of First Detection:</strong> {timestamp.strftime('%Y-%m-%d %H:%M:%S')}
                            <br>&bull; <strong>Total Detections in Event:</strong> {num_detections}
                            <br><br><strong>{attachment_msg}</strong>
                        </td></tr>
                        <tr><td style="padding: 20px 0;"></td></tr>
                        <tr><td align="center" style="padding-top: 20px; border-top: 1px solid {colors['border']}; font-family: Inter, Arial, sans-serif; color: #71717a; font-size: 12px;">This is an automated alert from the Argus Security System.</td></tr>
                    </table>
                </td></tr>
            </table>
        </body></html>"""
        return content, [permanent_path] if permanent_path else []

    def _send_email(self, subject: str, content: str, attachments: Optional[List[str]] = None):
        def email_task():
            yag = None
            try:
                logger.info(f"Sending email alert: {subject}")
                logger.debug(f"Recipients: {self.config.recipient_list}, Attachments: {attachments}")

                yag = yagmail.SMTP(user=self.config.sender_email, password=self.config.email_password)
                yag.send(to=self.config.recipient_list, subject=subject, contents=[content], attachments=attachments)

                logger.info(f"Email sent successfully to {self.config.recipient_list}")
                print(f"[EMAIL] Alert sent to {self.config.recipient_list}")
            except Exception as e:
                logger.error(f"Failed to send email: {e}")
                print(f"[EMAIL ERROR] {e}")
            finally:
                if yag:
                    yag.close()
        threading.Thread(target=email_task, daemon=True).start()

    def _extract_video_clip(self, video_path: str, max_size_mb: int = 40) -> Optional[str]:
        """Extract middle portion of video to fit within size limit.
        Returns path to extracted clip, or None if extraction fails."""

        if not video_path or not os.path.exists(video_path):
            logger.warning(f"Video file not found: {video_path}")
            return None

        try:
            # Get file size
            file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
            logger.info(f"Processing video: {video_path} ({file_size_mb:.2f}MB)")

            # If already under limit, return original (no ffmpeg needed)
            if file_size_mb <= max_size_mb:
                logger.info(f"Video is {file_size_mb:.2f}MB, under {max_size_mb}MB limit - sending original")
                return video_path

            # Video exceeds limit, verify source before extraction
            logger.info(f"Video {video_path} is {file_size_mb:.2f}MB, extracting middle portion")

            verify_source_cmd = [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'v:0',
                '-count_frames',
                '-show_entries', 'stream=nb_read_frames,codec_name',
                '-of', 'json',
                video_path
            ]

            try:
                verify_result = subprocess.run(verify_source_cmd, capture_output=True, text=True, timeout=10)
            except FileNotFoundError:
                logger.error("ffprobe not found - install ffmpeg and add to PATH for video extraction")
                logger.error("Sending original video without extraction (may fail if >50MB)")
                return video_path

            if verify_result.returncode != 0:
                logger.error(f"Source video validation failed: {verify_result.stderr}")
                logger.error("Source video appears to be corrupted - cannot send to Telegram")
                return None

            import json
            verify_data = json.loads(verify_result.stdout)
            if 'streams' in verify_data and len(verify_data['streams']) > 0:
                frame_count = verify_data['streams'][0].get('nb_read_frames', '0')
                codec = verify_data['streams'][0].get('codec_name', 'unknown')
                logger.info(f"Source video validation: {codec} codec, {frame_count} frames")

                if frame_count == '0' or int(frame_count) < 10:
                    logger.error(f"Source video has too few frames ({frame_count}) - video may be corrupted")
                    logger.error("This can happen if OpenCV's codec doesn't work properly with your system")
                    return None

            # Get video info using ffprobe
            ffprobe_cmd = [
                'ffprobe',
                '-v', 'error',
                '-show_entries', 'format=duration:stream=codec_name,width,height',
                '-of', 'json',
                video_path
            ]

            result = subprocess.run(ffprobe_cmd, capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                logger.error(f"ffprobe failed: {result.stderr}")
                return None

            import json
            probe_data = json.loads(result.stdout)

            # Get duration
            total_duration = float(probe_data['format']['duration'])

            # Log video properties for debugging
            if 'streams' in probe_data and len(probe_data['streams']) > 0:
                video_stream = probe_data['streams'][0]
                logger.info(f"Source video: {video_stream.get('codec_name')}, {video_stream.get('width')}x{video_stream.get('height')}, {total_duration:.1f}s")

            # Safety check: ensure video has meaningful duration
            if total_duration < 1.0:
                logger.error(f"Source video is too short ({total_duration:.2f}s) - cannot extract clip")
                return None

            # Calculate target duration based on size ratio
            # Assume linear relationship between duration and file size
            target_duration = (max_size_mb / file_size_mb) * total_duration
            # Apply safety margin (use 90% of calculated duration)
            target_duration = target_duration * 0.9

            # Calculate start time (middle of video, slightly biased toward beginning)
            # Detections typically peak early, so extract from 40% mark instead of 50%
            start_time = (total_duration - target_duration) * 0.4

            # Create output filename
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            output_path = os.path.join(os.path.dirname(video_path), f"{base_name}_clip.mp4")

            # Extract clip using ffmpeg
            # NOTE: Convert to H.264 for Telegram compatibility with forced keyframes
            ffmpeg_cmd = [
                'ffmpeg',
                '-y',  # Overwrite output file
                '-i', video_path,
                '-ss', str(start_time),  # Seek after input (accurate)
                '-t', str(target_duration),  # Duration
                '-c:v', 'libx264',  # Re-encode to H.264 for Telegram
                '-preset', 'medium',  # Encoding speed/quality tradeoff
                '-crf', '23',  # Quality (18-28 range, 23 is default)
                '-pix_fmt', 'yuv420p',  # Pixel format for compatibility
                '-g', '30',  # GOP size - keyframe every 30 frames
                '-sc_threshold', '0',  # Disable scene change detection
                '-force_key_frames', 'expr:gte(t,0)',  # Force keyframe at start
                '-c:a', 'aac',  # Re-encode audio with AAC
                '-b:a', '128k',  # Audio bitrate
                '-movflags', '+faststart',  # Enable streaming
                output_path
            ]

            logger.info(f"Extracting {target_duration:.1f}s clip from middle of {total_duration:.1f}s video")
            logger.debug(f"ffmpeg command: {' '.join(ffmpeg_cmd)}")
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=60)

            if result.returncode != 0:
                logger.error(f"ffmpeg extraction failed: {result.stderr}")
                return None

            # Verify output file exists and is valid
            if os.path.exists(output_path):
                output_size_mb = os.path.getsize(output_path) / (1024 * 1024)

                # Verify the extracted video is valid
                verify_cmd = [
                    'ffprobe',
                    '-v', 'error',
                    '-show_entries', 'format=duration',
                    '-of', 'default=noprint_wrappers=1:nokey=1',
                    output_path
                ]
                verify_result = subprocess.run(verify_cmd, capture_output=True, text=True, timeout=5)

                if verify_result.returncode == 0:
                    extracted_duration = float(verify_result.stdout.strip())
                    logger.info(f"Extracted clip: {output_size_mb:.2f}MB, {extracted_duration:.1f}s at {output_path}")

                    if extracted_duration < 0.5:
                        logger.error(f"Extracted clip is too short ({extracted_duration:.2f}s) - may be corrupted")
                        return None

                    return output_path
                else:
                    logger.error(f"Extracted clip verification failed: {verify_result.stderr}")
                    return None
            else:
                logger.error("ffmpeg completed but output file not found")
                return None

        except subprocess.TimeoutExpired:
            logger.error("Video extraction timed out")
            return None
        except Exception as e:
            logger.error(f"Error extracting video clip: {e}")
            return None

    def _prepare_telegram_message(self, alerts: List[Dict], video_path: Optional[str] = None) -> str:
        """Format alert message for Telegram."""
        viewport_name = alerts[0]['viewport_name']
        timestamp = datetime.fromtimestamp(alerts[0]['timestamp'])
        num_detections = len(alerts)
        primary_alert = max(alerts, key=lambda x: x['confidence'])

        # Build message with emoji and formatting
        message = f"🚨 *SECURITY ALERT*\n\n"
        message += f"*Camera:* {viewport_name}\n"
        message += f"*Time:* {timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n"
        message += f"*Confidence:* {primary_alert['confidence']}%\n"

        message += f"\n\n_Protected by Argus_"

        return message

    def _send_telegram_alert(self, message: str, screenshot_path: Optional[str] = None, video_path: Optional[str] = None) -> bool:
        """Send Telegram alert with screenshot and/or video. Returns True if successful, False otherwise."""
        if not self.telegram_enabled:
            logger.debug("Telegram not enabled")
            return False

        async def send_telegram_messages():
            """Async function to send Telegram messages."""
            # Create a fresh Bot instance for this thread/event loop with extended timeout
            from telegram.request import HTTPXRequest
            request = HTTPXRequest(
                connection_pool_size=8,
                read_timeout=60.0,      # 60 seconds for reading responses
                write_timeout=60.0,     # 60 seconds for uploading data
                connect_timeout=10.0    # 10 seconds for initial connection
            )
            bot = Bot(token=self.config.telegram_bot_token, request=request)

            success = False
            video_sent = False
            try:
                from io import BytesIO

                for chat_id in self.config.telegram_chat_ids:
                    try:
                        # Priority: Video > Photo > Text
                        # Send only ONE message with the alert content

                        if video_path and os.path.exists(video_path):
                            # Send video with caption (preferred if available)
                            with open(video_path, 'rb') as video_file:
                                video_data = video_file.read()

                            video_bytes = BytesIO(video_data)
                            video_bytes.name = os.path.basename(video_path)

                            # Get file size for logging
                            video_size_mb = len(video_data) / (1024 * 1024)
                            logger.info(f"Sending video clip ({video_size_mb:.2f}MB) to chat {chat_id}")

                            await bot.send_video(
                                chat_id=chat_id,
                                video=video_bytes,
                                caption=message,
                                parse_mode='Markdown',
                                supports_streaming=True
                            )
                            logger.info(f"Telegram video sent to chat {chat_id}")
                            print(f"[TELEGRAM] Video sent to chat {chat_id} ({video_size_mb:.2f}MB)")
                            video_sent = True

                        elif screenshot_path and os.path.exists(screenshot_path):
                            # Send photo with caption (if no video)
                            with open(screenshot_path, 'rb') as photo_file:
                                photo_data = photo_file.read()

                            photo_bytes = BytesIO(photo_data)
                            photo_bytes.name = os.path.basename(screenshot_path)

                            await bot.send_photo(
                                chat_id=chat_id,
                                photo=photo_bytes,
                                caption=message,
                                parse_mode='Markdown'
                            )
                            logger.info(f"Telegram photo sent to chat {chat_id}")

                        else:
                            # Send text message only if no media
                            await bot.send_message(
                                chat_id=chat_id,
                                text=message,
                                parse_mode='Markdown'
                            )
                            logger.info(f"Telegram message sent to chat {chat_id}")

                        print(f"[TELEGRAM] Alert sent to chat {chat_id}")
                        success = True

                    except TelegramError as e:
                        logger.error(f"Failed to send Telegram to chat {chat_id}: {e}")
                        print(f"[TELEGRAM ERROR] Chat {chat_id}: {e}")
                    except Exception as e:
                        logger.error(f"Unexpected error sending to chat {chat_id}: {e}")
                        print(f"[TELEGRAM ERROR] Chat {chat_id}: {e}")
            finally:
                # Clean up bot instance
                try:
                    await bot.shutdown()
                except:
                    pass

            return success, video_sent

        def telegram_task():
            """Background task wrapper."""
            try:
                logger.info("Sending Telegram alert...")
                logger.debug(f"Chat IDs: {self.config.telegram_chat_ids}, Screenshot: {screenshot_path is not None}, Video: {video_path is not None}")

                # Use asyncio.run() which properly manages event loop lifecycle
                success, video_sent = asyncio.run(send_telegram_messages())

                # Clean up temporary video clip after successful send
                if video_sent and video_path and video_path.endswith('_clip.mp4'):
                    try:
                        if os.path.exists(video_path):
                            os.remove(video_path)
                            logger.info(f"Deleted temporary video clip: {video_path}")
                            print(f"[CLEANUP] Deleted telegram clip: {os.path.basename(video_path)}")
                    except Exception as e:
                        logger.error(f"Failed to delete temporary clip {video_path}: {e}")

            except Exception as e:
                logger.error(f"Failed to send Telegram alert: {e}")
                print(f"[TELEGRAM ERROR] {e}")

        # Run in background thread
        thread = threading.Thread(target=telegram_task, daemon=True)
        thread.start()

        # Return True to indicate attempt was made (actual success is logged)
        return True

    def stop(self):
        """Stop alert manager and flush pending database writes."""
        with self.alert_lock:
            for timer in self.batch_timers.values():
                if timer:
                    timer.cancel()

        # Stop database worker and flush remaining events
        self.db_worker_running = False
        if self.db_worker_thread.is_alive():
            self.db_worker_thread.join(timeout=2.0)

class Viewport:
    def __init__(self, viewport_id: tuple, config: dict):
        self.id = viewport_id
        self.config = ViewportConfig(**config)
        
        motion_config = MotionDetectorConfig(self.config.sensitivity, self.config.motion_aggressiveness)
        self.motion_detector = MotionDetector(motion_config, self.id)
        
        self.frame_counter = 0
        self.last_detection_timestamp = 0
        self.detection_timeout_duration = 5.0

    def process_frame(self, frame: np.ndarray, yolo_manager: YOLOProcessManager) -> np.ndarray:
        """Process frame and submit to YOLO manager if needed.
        Frame is already scaled by scale_factor before reaching this method."""
        self.frame_counter += 1

        # Frame is already scaled - no need to scale again
        motion_detected = self.motion_detector.check(frame)

        now = time.time()
        is_person_currently_tracked = (now - self.last_detection_timestamp) < self.detection_timeout_duration
        should_run_yolo = (motion_detected or is_person_currently_tracked) and (self.frame_counter % self.config.yolo_interval == 0)

        if should_run_yolo:
            try:
                task_id = yolo_manager.submit_detection_task(self.id, frame)
            except Full:
                pass # Drop frame if detection is backlogged

        return frame
        
    def update_config(self, new_config_data: dict):
        """Updates the viewport's configuration dynamically."""
        current_config = asdict(self.config)
        current_config.update(new_config_data)
        self.config = ViewportConfig(**current_config)
        
        # Re-initialize the motion detector with new sensitivity settings
        motion_config = MotionDetectorConfig(self.config.sensitivity, self.config.motion_aggressiveness)
        self.motion_detector = MotionDetector(motion_config, self.id)
        print(f"Configuration updated for viewport {self.id}")

class SecuritySystem:
    def __init__(self, config_data: dict, socketio_instance: SocketIO, flask_app=None):
        self.config = config_data
        self.socketio = socketio_instance
        self.flask_app = flask_app  # Store Flask app for application context
        self.running = False

        # Calculate dynamic grid size based on enabled cameras
        enabled_cameras = [cam for cam in self.config['cameras'] if cam.get('enabled', True)]
        self.grid_size = calculate_optimal_grid(len(enabled_cameras))

        # Create dynamic viewport configurations
        self.viewport_configs = create_viewport_configs_from_cameras(
            self.config['cameras'],
            self.config['viewport_defaults']
        )

        # --- FRAME BROKERING SYSTEM ---
        # Initialize global memory manager
        total_memory_limit = self.config.get('system', {}).get('memory_limit_mb', 1024)
        self.memory_manager = GlobalMemoryManager(total_memory_limit)

        # Initialize shared frame pool for efficient memory usage
        # Full-resolution 1080p frames (6.2MB each)
        self.frame_pool = SharedFramePool(max_memory_mb=512)

        # Initialize frame broker
        broker_memory_limit = min(512, total_memory_limit // 4)  # Use quarter of total for broker
        self.frame_broker = FrameBroker(max_memory_mb=broker_memory_limit, memory_manager=self.memory_manager)

        # Initialize metrics system
        self.metrics = get_global_metrics()

        # --- CAMERA MANAGEMENT ---
        camera_configs = create_camera_configs(self.config)
        self.camera_manager = CameraManager(camera_configs, self.frame_broker)

        # Get actual camera FPS for video recording (use first enabled camera's FPS)
        self.recording_fps = camera_configs[0].fps if camera_configs else self.config['system']['max_fps']

        # --- ALERT QUEUE ---
        self.alert_queue = Queue(maxsize=10)

        # --- SHARED COMPONENTS ---
        self.yolo_manager = YOLOProcessManager(
            model_path=self.config['system']['model_path'],
            device=self.config['system'].get('yolo_device'),
            max_queue_size=20
        )

        # --- MEMORY-BOUNDED FRAME BUFFERS ---
        self.viewport_buffers: Dict[tuple, MemoryBoundedBuffer] = {}
        self.recording_buffers: Dict[tuple, MemoryBoundedBuffer] = {}
        self.display_buffers: Dict[tuple, MemoryBoundedBuffer] = {}

        # Calculate memory allocation per viewport to stay within global limit
        num_viewports = len(self.viewport_configs)

        # Viewport buffer: Stores scaled frames (~1MB with scale_factor=0.4)
        # Recording buffer: Stores full-res frames (~6.2MB) for video recording
        # Display buffer: Only needs 1-2 latest frames for JPEG encoding
        viewport_memory_mb = 30   # ~30 scaled frames for motion detection
        recording_memory_mb = 50  # ~8 full-res frames for pre-event recording
        display_memory_mb = 7     # ~1 full-res frame for web display (minimal)

        logger.info(f"Memory allocation: {broker_memory_limit}MB broker, {num_viewports} viewports with {viewport_memory_mb}MB/{recording_memory_mb}MB/{display_memory_mb}MB buffers")

        # Create memory-bounded buffers for each viewport
        for viewport_id_str in self.viewport_configs.keys():
            r, c = map(int, viewport_id_str.split(','))
            vp_id = (r, c)

            # Create buffers with different eviction policies
            self.viewport_buffers[vp_id] = MemoryBoundedBuffer(
                max_memory_bytes=viewport_memory_mb * 1024 * 1024,
                eviction_policy=EvictionPolicy.PRIORITY,
                name=f"viewport_{r}_{c}"
            )

            self.recording_buffers[vp_id] = MemoryBoundedBuffer(
                max_memory_bytes=recording_memory_mb * 1024 * 1024,
                eviction_policy=EvictionPolicy.FIFO,
                name=f"recording_{r}_{c}"
            )

            self.display_buffers[vp_id] = MemoryBoundedBuffer(
                max_memory_bytes=display_memory_mb * 1024 * 1024,
                eviction_policy=EvictionPolicy.LRU,
                name=f"display_{r}_{c}"
            )

            # Register buffers with memory manager
            self.memory_manager.register_buffer(self.viewport_buffers[vp_id])
            self.memory_manager.register_buffer(self.recording_buffers[vp_id])
            self.memory_manager.register_buffer(self.display_buffers[vp_id])

        # Initialize Viewports from dynamic configurations
        self.viewports: Dict[tuple, Viewport] = {}
        for viewport_id_str, vp_config in self.viewport_configs.items():
            r, c = map(int, viewport_id_str.split(','))
            vp_id = (r, c)
            self.viewports[vp_id] = Viewport(vp_id, vp_config)

        shared_lock = threading.RLock()

        # Initialize components and pass the shared lock to them
        rec_conf = self.config['video_recording']

        # Use default frame dimensions - will be updated when cameras connect
        default_width, default_height = 640, 480
        vp_width = default_width // self.grid_size[1] if self.grid_size[1] > 0 else default_width
        vp_height = default_height // self.grid_size[0] if self.grid_size[0] > 0 else default_height

        # Use actual camera FPS for video recording (not max_fps) to match frame arrival rate
        self.video_recorder = VideoClipRecorder(rec_conf, vp_width, vp_height, self.recording_fps, shared_lock)
        
        am_conf = self.config['alert_manager']
        alert_config_obj = AlertConfig(
            curfew_start=datetime.strptime(am_conf['curfew_start'], '%H:%M').time(),
            curfew_end=datetime.strptime(am_conf['curfew_end'], '%H:%M').time(),
            cooldown_period=am_conf['cooldown_period_seconds'],
            batch_window=am_conf['batch_window_seconds'],
            recipient_list=am_conf['recipient_list'],
            sender_email=am_conf.get('sender_email'),
            sender_name=am_conf.get('sender_name'),
            email_password=am_conf.get('email_password'),
            # Telegram configuration
            telegram_bot_token=am_conf.get('telegram_bot_token'),
            telegram_chat_ids=am_conf.get('telegram_chat_ids'),
            telegram_enabled=am_conf.get('telegram_enabled', False),
            email_as_fallback=am_conf.get('email_as_fallback', True)
        )
        self.alert_manager = AlertManager(alert_config_obj, self.socketio, shared_lock, self.video_recorder, self.flask_app)
        
        # --- FRAME BROKER SETUP ---
        # Register cameras with frame broker
        for camera_config in camera_configs:
            if camera_config.enabled:
                self.frame_broker.register_camera(
                    camera_config.camera_name,
                    base_fps=camera_config.fps
                )

                # Subscribe to camera frames for processing
                self.frame_broker.subscribe_to_camera(
                    camera_config.camera_name,
                    self._process_camera_frame
                )

        # --- WORKER THREADS ---
        self.worker_threads = []

        # Track if video dimensions have been updated from actual frames
        self._video_dimensions_updated = False

        # --- JPEG ENCODING FOR WEB STREAMING ---
        # Pre-encoded JPEG frames for web clients (avoid encoding per client)
        self.encoded_frames: Dict[tuple, bytes] = {}
        self.encoding_lock = threading.Lock()

    def start(self):
        self.running = True

        # Start cameras first
        if not self.camera_manager.start_all():
            print("[WARNING] No cameras started successfully")
            return False

        # Start YOLO process
        if not self.yolo_manager.start_process():
            print("[ERROR] Failed to start YOLO process. Exiting.")
            self.camera_manager.stop_all()
            return False

        # Create all singleton worker threads inside list
        self.worker_threads = [
            threading.Thread(target=self._metrics_worker, daemon=True),  # Metrics collection
            threading.Thread(target=self._yolo_result_processor_worker, daemon=True), # YOLO processor
            threading.Thread(target=self._event_handler_worker, daemon=True), # Event handler
            threading.Thread(target=self._jpeg_encoding_worker, daemon=True),  # Centralized JPEG encoding
        ]

        # Create viewport workers according to number of viewports
        num_viewports = len(self.viewport_configs)

        # Use 1 worker per 2 viewports, min 1, max CPU_count/2
        num_viewport_workers = max(1, min(num_viewports // 2, os.cpu_count() // 2))
        logger.info(f"Creating {num_viewport_workers} viewport workers for {num_viewports} viewports")

        # Create and add to list
        for i in range(num_viewport_workers):
            worker = threading.Thread(
                target=self._viewport_worker,
                daemon=True,
                name=f"ViewportWorker-{i+1}"
            )
            self.worker_threads.append(worker)

        # Start all threads
        for thread in self.worker_threads:
            thread.start()
        print("[STARTUP] SecuritySystem processing pipeline started.")
        return True

    def stop(self):
        self.running = False

        self.camera_manager.stop_all()
        self.yolo_manager.stop_process()
        self.frame_broker.shutdown()

        for thread in self.worker_threads:
            thread.join(timeout=2.0)
        self.alert_manager.stop()
        self.metrics.shutdown()
        self.frame_pool.shutdown()
        print("[OK] SecuritySystem stopped.")

    def _process_camera_frame(self, camera_name: str, frame: np.ndarray, timestamp: float) -> None:
        """Callback: Process frame from camera via frame broker."""
        try:
            # Record metrics (reduced frequency to avoid overhead)
            if timestamp % 5.0 < 0.04:  # Log approximately once per 5 seconds
                record_metric("camera_fps", 1.0, {"camera": camera_name})
                record_metric("frames_per_second", 1.0)

            # Process viewports that use this camera
            for vp_id, viewport in self.viewports.items():
                if viewport.config.camera_name == camera_name:
                    viewport_frame = self._extract_viewport_frame(frame, vp_id)
                    if viewport_frame is not None:
                        # Update video recorder dimensions on first frame
                        if not self._video_dimensions_updated:
                            height, width = viewport_frame.shape[:2]
                            self.video_recorder.update_dimensions(width, height)
                            self._video_dimensions_updated = True

                        # Add full-res frame to display buffer for responsive web UI
                        # Buffer will auto-evict old frames (FIFO) - only latest frame needed
                        if vp_id in self.display_buffers:
                            if not self.display_buffers[vp_id].add_frame(viewport_frame, timestamp, 0):
                                # This is expected when buffer is full - just replace old frame
                                pass

                        # Determine priority based on motion detection state
                        priority = 1 if vp_id in getattr(viewport, '_motion_active_viewports', set()) else 0

                        # Apply scale factor before adding to processing pool (save memory)
                        viewport_config = self.viewport_configs.get(f"{vp_id[0]},{vp_id[1]}", {})
                        scale_factor = viewport_config.get('scale_factor', 1.0)
                        if scale_factor < 1.0 and scale_factor > 0:
                            h, w = viewport_frame.shape[:2]
                            # Cache scaled dimensions to avoid repeated int conversions
                            cache_key = (w, h, scale_factor)
                            if not hasattr(self, '_scale_cache'):
                                self._scale_cache = {}

                            if cache_key not in self._scale_cache:
                                self._scale_cache[cache_key] = (int(w * scale_factor), int(h * scale_factor))

                            scaled_w, scaled_h = self._scale_cache[cache_key]

                            # Use INTER_NEAREST for speed (vs default INTER_LINEAR)
                            # ~3x faster, acceptable quality loss for motion detection
                            processing_frame = cv2.resize(viewport_frame, (scaled_w, scaled_h), interpolation=cv2.INTER_NEAREST)
                        else:
                            processing_frame = viewport_frame

                        # Add scaled frame to shared pool once and distribute references
                        frame_ref = self.frame_pool.add_frame(processing_frame, timestamp)

                        if frame_ref:
                            # Add reference to viewport buffer for motion detection processing
                            if vp_id in self.viewport_buffers:
                                viewport_ref = self.frame_pool.acquire_reference(frame_ref.frame_id)
                                if viewport_ref:
                                    if not self.viewport_buffers[vp_id].add_frame(viewport_ref, timestamp, priority):
                                        viewport_ref.release()
                                        record_metric("frame_drop_rate", 1.0, {"reason": "viewport_buffer_full", "viewport": f"{vp_id[0]},{vp_id[1]}"})

                            # Add reference to recording buffer (resize for video recorder dimensions)
                            if vp_id in self.recording_buffers:
                                # Ensure frame matches video recorder dimensions
                                if viewport_frame.shape[:2] != (self.video_recorder.frame_height, self.video_recorder.frame_width):
                                    # Use INTER_NEAREST for speed (acceptable quality for recording)
                                    recording_frame = cv2.resize(viewport_frame, (self.video_recorder.frame_width, self.video_recorder.frame_height), interpolation=cv2.INTER_NEAREST)
                                else:
                                    recording_frame = viewport_frame

                                # Store resized frame directly (not using pool for recording buffer due to resize)
                                if not self.recording_buffers[vp_id].add_frame(recording_frame, timestamp, priority):
                                    record_metric("frame_drop_rate", 1.0, {"reason": "recording_buffer_full", "viewport": f"{vp_id[0]},{vp_id[1]}"})

                            # Release original reference (buffers have their own)
                            frame_ref.release()
                        else:
                            # Pool is full - drop frame
                            record_metric("frame_drop_rate", 1.0, {"reason": "frame_pool_full", "viewport": f"{vp_id[0]},{vp_id[1]}"})

                        # Feed frames to active recordings
                        if self.video_recorder.is_recording(vp_id):
                            with self.video_recorder.lock:
                                state = self.video_recorder.recording_state.get(vp_id)
                                if state and not state['done'].is_set():
                                    try:
                                        state['frame_queue'].put_nowait(viewport_frame)
                                    except Full:
                                        pass  # Drop frame if queue is full

        except Exception as e:
            logger.error(f"Error processing camera frame from {camera_name}: {e}")
            record_metric("frame_processing_errors", 1.0, {"camera": camera_name})

    def _metrics_worker(self):
        """Worker: Collect and update system metrics."""
        last_metrics_update = time.time()
        metrics_interval = 5.0  # Update every 5 seconds

        while self.running:
            try:
                current_time = time.time()
                if current_time - last_metrics_update >= metrics_interval:
                    # Collect memory metrics
                    memory_stats = self.memory_manager.get_global_stats()
                    record_metric("memory_usage_mb", memory_stats['total_memory_usage_mb'])
                    record_metric("memory_pressure_level", memory_stats['memory_pressure_level'] * 100)

                    # Collect camera metrics
                    camera_status = self.frame_broker.get_camera_status()
                    for camera_name, status in camera_status.items():
                        record_metric("camera_frame_drops", status['dropped_frames'], {"camera": camera_name})
                        record_metric("camera_fps", status['current_fps'], {"camera": camera_name})

                    # Handle memory pressure
                    if self.memory_manager.is_memory_pressure_critical():
                        logger.warning("Critical memory pressure detected - attempting cleanup")
                        if not self.memory_manager.handle_memory_pressure():
                            logger.error("Failed to resolve critical memory pressure")

                    last_metrics_update = current_time

                time.sleep(1.0)  # Check every second

            except Exception as e:
                logger.error(f"Error in metrics worker: {e}")
                time.sleep(5.0)  # Back off on error

    def _extract_viewport_frame(self, camera_frame: np.ndarray, viewport_id: tuple) -> Optional[np.ndarray]:
        """Extract viewport frame - returns full camera frame (no virtual viewports)."""
        # Each camera outputs full resolution, no splitting needed
        return camera_frame

    def _viewport_worker(self):
        """Worker: Event-driven frame processing for motion detection."""
        # Assign viewports to this worker using simple round-robin
        worker_name = threading.current_thread().name

        # Extract worker number from name (e.g., "ViewportWorker-1" -> 1)
        try:
            worker_num = int(worker_name.split('-')[1])
        except:
            worker_num = 1

        all_viewport_ids = sorted(list(self.viewport_buffers.keys()))

        # Simple modulo-based assignment: worker N gets viewports where index % total_workers == N-1
        total_workers = sum(1 for t in self.worker_threads if 'ViewportWorker' in t.name)
        assigned_viewports = [vp for i, vp in enumerate(all_viewport_ids) if i % total_workers == (worker_num - 1)]

        logger.info(f"Viewport worker '{worker_name}' assigned to viewports: {assigned_viewports} (worker {worker_num}/{total_workers})")

        frames_processed = 0
        while self.running:
            try:
                # Process frames from assigned viewports using event-driven approach
                for vp_id in assigned_viewports:
                    if vp_id not in self.viewport_buffers:
                        continue

                    buffer = self.viewport_buffers[vp_id]
                    viewport = self.viewports.get(vp_id)
                    if viewport is None:
                        continue

                    # Wait for frame to be available (blocks until notified or timeout)
                    # Timeout allows cycling through assigned viewports and checking self.running
                    frame_entry = buffer.wait_and_pop_entry(timeout=0.05)
                    if frame_entry is None:
                        continue

                    try:
                        # The frame attribute of FrameEntry can be a FrameReference or np.ndarray
                        frame_obj = frame_entry.frame
                        frame_data = frame_entry.get_frame_data()

                        if frame_data is not None:
                            frames_processed += 1
                            if frames_processed % 100 == 0:
                                logger.info(f"Viewport worker '{worker_name}' processed {frames_processed} frames for {vp_id}")

                            # Record processing metrics
                            start_time = time.time()

                            # Process the frame for motion detection
                            viewport.process_frame(frame_data, self.yolo_manager)

                            processing_time = (time.time() - start_time) * 1000
                            record_metric("processing_latency_ms", processing_time, {"viewport": f"{vp_id[0]},{vp_id[1]}"})
                    finally:
                        # FrameEntry handles releasing the reference if it is one
                        frame_entry.release()

                    # Update frame broker priority based on motion detection
                    camera_name = viewport.config.camera_name
                    priority = 1 if hasattr(viewport, 'last_detection_timestamp') and \
                                  (time.time() - viewport.last_detection_timestamp) < 10 else 0
                    self.frame_broker.set_camera_priority(camera_name, priority)

                    # Display buffer is updated in frame broker callback with full-res frames

            except Exception as e:
                logger.error(f"Error in viewport worker: {e}")
                record_metric("viewport_processing_errors", 1.0)

    def _yolo_result_processor_worker(self):
        """Worker: Processes results from the YOLO subprocess."""
        results_processed = 0
        last_log_time = time.time()

        # DATA COLLECTION FOR REPORT
        inference_times = []

        while self.running:
            try:
                # Get results from YOLO process (non-blocking)
                results = self.yolo_manager.get_detection_results(max_results=10)

                # DATA COLLECTION FOR REPORT
                for viewport_id, detections, inference_time in results:
                    inference_times.append(inference_time)
                    # Log every 50th inference
                    if len(inference_times) % 50 == 0:
                        avg = sum(inference_times) / len(inference_times)
                        p95 = sorted(inference_times)[int(len(inference_times) * 0.95)]

                        print(f"[REPORT] Inference Stats (n={len(inference_times)}):+  Avg: {avg*1000:.1f}ms, 95th: {p95*1000:.1f}ms")  

                #END DATA COLLECTION
                                
                if not results:
                    time.sleep(0.01)  # Small delay if no results
                    continue
                
                for viewport_id, detections, inference_time in results:
                    results_processed += 1

                    viewport = self.viewports.get(viewport_id)
                    if not viewport:
                        print(f"[WARNING] Received results for unknown viewport: {viewport_id}")
                        continue
                        
                    # Debug logging
                    if results_processed % 10 == 0:
                        print(f"[YOLO] Processing YOLO result {results_processed} for {viewport_id}: "
                            f"{len(detections)} detections, {inference_time:.3f}s inference")
                        for i, det in enumerate(detections[:3]):  # Show first 3 detections
                            print(f"   Detection {i}: confidence={det['confidence']:.3f}, "
                                f"box={det['box']}")

                    # Filter detections by confidence
                    person_detections = [d for d in detections
                                    if d['confidence'] >= viewport.config.min_confidence]

                    if person_detections:
                        viewport.last_detection_timestamp = time.time()
                        
                        # Create event and submit to alert queue
                        primary_detection = max(person_detections, key=lambda p: p['confidence'])
                        event = {
                            'viewport_id': viewport_id,
                            'viewport_name': viewport.config.camera_name,
                            'timestamp': time.time(),
                            'confidence': int(primary_detection['confidence'] * 100),
                            'all_detections': person_detections,
                            'inference_time': inference_time,
                            'frame': None  # We don't have the original frame here
                        }
                        
                        try:
                            self.alert_queue.put_nowait(event)
                            print(f"[OK] Alert event queued for {viewport_id}")
                        except Full:
                            print(f"[WARNING] Alert queue full, dropping event for {viewport_id}")
                
                # Periodic cleanup and performance logging
                current_time = time.time()
                if current_time - last_log_time > 30.0:  # Every 30 seconds
                    self.yolo_manager.cleanup_stale_tasks()
                    stats = self.yolo_manager.get_performance_stats()
                    if stats['total_inferences'] > 0:
                        print(f"[PERF] YOLO Performance: {stats['avg_inference_time']:.3f}s avg, "
                            f"{stats['total_inferences']} total, "
                            f"{stats['pending_tasks']} pending")
                    print(f"[PERF] Results processed in last 30s: {results_processed}")
                    results_processed = 0
                    last_log_time = current_time

            except Exception as e:
                print(f"[ERROR] Error in YOLO result processor: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)

    def _event_handler_worker(self):
        """Worker: Listens to the alert queue and triggers recorder and alert manager."""
        while self.running:
            try:
                event = self.alert_queue.get(timeout=1.0)

                if not self.alert_manager.detection_enabled:
                    continue # Ignore event and wait for next alert.

                # If the event doesn't have a frame, get the latest one from the display buffer.
                # This is the "zero-copy" part - we retrieve the frame in the main process.
                viewport_id = event['viewport_id']
                if event.get('frame') is None and viewport_id in self.display_buffers:
                    current_frame = self.display_buffers[viewport_id].get_latest_frame()
                    if current_frame is not None:
                        logger.debug(f"Retrieved latest frame for event in viewport {viewport_id}")
                        event['frame'] = current_frame

                # Process video recording and alerts
                recording_buffer = self.recording_buffers.get(viewport_id)
                video_path = self.video_recorder.handle_alert_event(event, recording_buffer)
                event['video_path'] = video_path
                self.alert_manager.handle_alert_event(event)

            except Empty:
                continue

    def _jpeg_encoding_worker(self):
        """Worker: Pre-encodes frames to JPEG for web streaming to avoid per-client encoding."""
        encoding_quality = 60  # Reduce for better performance
        target_fps = 25
        frame_interval = 1.0 / target_fps

        frames_encoded = 0
        last_log_time = time.time()

        while self.running:
            try:
                start_time = time.time()

                # Encode frames for all viewports
                encoded_this_cycle = 0
                for vp_id, display_buffer in self.display_buffers.items():
                    frame = display_buffer.get_latest_frame()
                    if frame is not None:
                        try:
                            # Scale down for web streaming if very large (reduces encoding time & bandwidth)
                            h, w = frame.shape[:2]
                            max_dimension = 1280  # Max width/height for web display
                            if w > max_dimension or h > max_dimension:
                                scale = max_dimension / max(w, h)
                                new_w = int(w * scale)
                                new_h = int(h * scale)
                                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

                            # Encode frame to JPEG with optimized parameters
                            encode_params = [
                                cv2.IMWRITE_JPEG_QUALITY, encoding_quality,
                                cv2.IMWRITE_JPEG_OPTIMIZE, 1  # Enable JPEG optimization
                            ]
                            _, buffer = cv2.imencode('.jpg', frame, encode_params)
                            encoded_frame = buffer.tobytes()

                            # Store encoded frame
                            with self.encoding_lock:
                                self.encoded_frames[vp_id] = encoded_frame

                            frames_encoded += 1
                            encoded_this_cycle += 1

                            record_metric("jpeg_encoding_ms", (time.time() - start_time) * 1000,
                                        {"viewport": f"{vp_id[0]},{vp_id[1]}"})
                        except Exception as e:
                            logger.error(f"Error encoding frame for viewport {vp_id}: {e}")

                # Log encoding stats every 5 seconds
                if time.time() - last_log_time > 5.0:
                    logger.info(f"JPEG encoder: {frames_encoded} total frames encoded, {encoded_this_cycle} this cycle")
                    last_log_time = time.time()

                # Maintain target FPS for encoding
                elapsed = time.time() - start_time
                sleep_time = max(0, frame_interval - elapsed)
                time.sleep(sleep_time)

            except Exception as e:
                logger.error(f"Error in JPEG encoding worker: {e}")
                time.sleep(0.1)

    def get_encoded_frame(self, viewport_id: tuple) -> Optional[bytes]:
        """Get pre-encoded JPEG frame for a viewport."""
        with self.encoding_lock:
            return self.encoded_frames.get(viewport_id)    


    def get_viewport(self, vp_id: tuple) -> Optional[Viewport]:
        return self.viewports.get(vp_id)
    
    def get_yolo_performance_stats(self) -> Dict[str, float]:
        """Get YOLO process performance statistics."""
        return self.yolo_manager.get_performance_stats()
    
    def get_camera_status(self) -> Dict[str, Dict]:
        """Get status of all cameras."""
        return self.camera_manager.get_camera_status()
    
    def restart_camera(self, camera_name: str) -> bool:
        """Restart a specific camera."""
        return self.camera_manager.restart_camera(camera_name)