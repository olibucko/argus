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
from typing import Tuple, List, Optional, Dict, Any

import cv2
import numpy as np
import yagmail
from flask_socketio import SocketIO
import sqlite3

from .yolo_process import YOLOProcessManager
from .motion_detector import MotionDetector, MotionDetectorConfig
from .camera_manager import CameraManager, create_camera_configs
from .frame_broker import FrameBroker
from .memory_manager import MemoryBoundedBuffer, GlobalMemoryManager, EvictionPolicy
from .frame_pool import SharedFramePool
from .metrics import get_global_metrics, record_metric

logger = logging.getLogger(__name__)

try:
    from telegram import Bot
    from telegram.error import TelegramError
    from telegram.request import HTTPXRequest
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    logger.warning("python-telegram-bot not installed. Telegram alerts will be disabled.")


#### CONSTANTS ###
DB_FILE = "security_events.db"
MAX_CONCURRENT_RECORDINGS = 4

### HELPER FUNCTIONS ###
def calculate_optimal_grid(num_cameras: int) -> Tuple[int, int]:
    """Calculate optimal grid dimensions for the given number of cameras."""
    if num_cameras <= 0: return (1, 1)
    if num_cameras == 1: return (1, 1)
    if num_cameras <= 4: return (2, 2)
    if num_cameras <= 6: return (2, 3)
    if num_cameras <= 9: return (3, 3)
    if num_cameras <= 12: return (3, 4)
    if num_cameras <= 16: return (4, 4)
    if num_cameras <= 20: return (4, 5)
    cols = int(np.ceil(np.sqrt(num_cameras * 1.6)))
    rows = int(np.ceil(num_cameras / cols))
    return (rows, cols)

def create_viewport_configs_from_cameras(cameras: List[Dict], viewport_defaults: Dict) -> Dict[str, Dict]:
    """Create viewport configurations from camera list with dynamic grid positioning."""
    enabled_cameras = [cam for cam in cameras if cam.get('enabled', True)]
    if not enabled_cameras: return {}

    rows, cols = calculate_optimal_grid(len(enabled_cameras))
    viewport_configs = {}

    for i, camera in enumerate(enabled_cameras):
        row, col = i // cols, i % cols
        viewport_id = f"{row},{col}"
        config = viewport_defaults.copy()
        # Merge camera settings into defaults
        for k, v in config.items():
            if k in camera:
                config[k] = camera[k]
        
        config['camera_name'] = camera.get('camera_name')
        config['name'] = camera.get('name', camera.get('camera_name'))
        viewport_configs[viewport_id] = config
        
    return viewport_configs

### DATACLASSES ###
@dataclass
class ViewportConfig:
    camera_name: str = "Unnamed"
    name: str = "Unnamed"
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
    sender_email: Optional[str] = None
    sender_name: Optional[str] = None
    email_password: Optional[str] = None
    telegram_bot_token: Optional[str] = None
    telegram_chat_ids: Optional[List[str]] = None
    telegram_enabled: bool = False
    email_as_fallback: bool = True

### CORE CLASSES ###
class VideoClipRecorder:
    """Handles pre- and post-event video recording for detected security events."""
    def __init__(self, config: Dict, frame_width: int, frame_height: int, fps: int, shared_lock: threading.RLock) -> None:
        self.video_dir = "event_videos"
        os.makedirs(self.video_dir, exist_ok=True)
        self.pre_event_seconds: int = config['pre_event_seconds']
        self.post_event_seconds: int = config['post_event_seconds']
        self.max_clip_duration: int = config['max_clip_duration_seconds']
        self.codec: int = cv2.VideoWriter_fourcc(*config['codec'])
        self.frame_width, self.frame_height, self.fps = frame_width, frame_height, fps
        self._validate_codec()
        self.recording_state: Dict[Tuple[int, int], Dict] = {}
        self.lock: threading.RLock = shared_lock

    def _validate_codec(self) -> None:
        """Tests that the configured video codec is usable."""
        test_path = "test_codec.mp4"
        test_writer = cv2.VideoWriter(test_path, self.codec, self.fps, (self.frame_width, self.frame_height))
        if not test_writer.isOpened():
            logger.error(f"Codec FourCC '{self.codec}' is not compatible. Video recording will fail.")
        else:
            logger.info(f"Video codec validated successfully.")
        test_writer.release()
        if os.path.exists(test_path):
            os.remove(test_path)

    def update_dimensions(self, frame_width: int, frame_height: int) -> None:
        """Update the video recorder dimensions based on actual frame sizes."""
        self.frame_width, self.frame_height = frame_width, frame_height
        logger.info(f"VideoClipRecorder dimensions updated to {frame_width}x{frame_height}")

    def is_recording(self, viewport_id: Tuple[int, int]) -> bool:
        """Checks if a recording is currently active for a given viewport."""
        with self.lock:
            state = self.recording_state.get(viewport_id)
            return state and not state['done'].is_set()

    def update_activity(self, viewport_id: Tuple[int, int]) -> None:
        """Updates the last activity timestamp for an active recording."""
        with self.lock:
            if state := self.recording_state.get(viewport_id):
                state['last_activity_time'] = time.time()

    def handle_alert_event(self, event: Dict, recording_buffer: MemoryBoundedBuffer) -> Optional[str]:
        """Checks recording state and starts or updates a recording based on the event."""
        viewport_id = event['viewport_id']
        if self.is_recording(viewport_id):
            self.update_activity(viewport_id)
            return self.recording_state[viewport_id]['filepath']
        
        timestamp = datetime.fromtimestamp(event['timestamp'])
        return self.start_recording(viewport_id, timestamp, event['viewport_name'], recording_buffer)

    def start_recording(self, viewport_id: Tuple[int, int], timestamp: datetime, viewport_name: str, recording_buffer: MemoryBoundedBuffer) -> Optional[str]:
        """Starts a new video recording thread for a given viewport."""
        with self.lock:
            if self.is_recording(viewport_id):
                self.update_activity(viewport_id)
                return self.recording_state[viewport_id]['filepath']
            if len(self.recording_state) >= MAX_CONCURRENT_RECORDINGS:
                logger.warning(f"Max concurrent recordings ({MAX_CONCURRENT_RECORDINGS}) reached.")
                return None

            filename = f"{timestamp.strftime('%Y%m%d_%H%M%S')}_{viewport_name.replace(' ', '_')}.mp4"
            filepath = os.path.join(self.video_dir, filename)
            
            self.recording_state[viewport_id] = {
                'filepath': filepath,
                'done': threading.Event(),
                'start_time': time.time(),
                'last_activity_time': time.time(),
                'frame_queue': Queue(maxsize=int(self.max_clip_duration * self.fps * 1.2))
            }
            
            thread = threading.Thread(target=self._write_video_thread, args=(viewport_id, filepath, recording_buffer.get_all_frames()))
            thread.daemon = True
            thread.start()
            return filepath

    def _write_video_thread(self, viewport_id: Tuple[int, int], filepath: str, pre_event_frames: List[np.ndarray]) -> None:
        """The target function for the video writing thread."""
        with self.lock:
            state = self.recording_state.get(viewport_id)
        
        if not state:
            logger.error(f"Could not find recording state for {viewport_id}")
            return

        out = cv2.VideoWriter(filepath, self.codec, self.fps, (self.frame_width, self.frame_height))
        if not out.isOpened():
            logger.error(f"Failed to open video writer for {filepath}")
            return

        for frame in pre_event_frames:
            out.write(frame)

        while True:
            now = time.time()
            if (now - state['last_activity_time'] > self.post_event_seconds) or (now - state['start_time'] > self.max_clip_duration):
                break
            try:
                frame = state['frame_queue'].get(timeout=1.0)
                out.write(frame)
            except Empty:
                continue

        while not state['frame_queue'].empty():
            try:
                frame = state['frame_queue'].get_nowait()
                out.write(frame)
            except Empty:
                break

        out.release()
        logger.info(f"Saved video clip to {filepath}")
        with self.lock:
            state['done'].set()
            self.recording_state.pop(viewport_id, None)

class AlertManager:
    """Manages the security alert lifecycle from detection to notification."""
    def __init__(self, config: AlertConfig, socketio: SocketIO, lock: threading.RLock, recorder: VideoClipRecorder, app: Optional[Any] = None) -> None:
        self.config = config
        self.socketio = socketio
        self.flask_app = app
        self.alert_lock = lock
        self.video_recorder = recorder
        self.last_alert_times: Dict[tuple, float] = defaultdict(float)
        self.pending_alerts: Dict[tuple, List[Dict]] = defaultdict(list)
        self.batch_timers: Dict[tuple, Optional[threading.Timer]] = {}
        os.makedirs("event_captures", exist_ok=True)
        
        self.detection_enabled = False
        self.telegram_enabled = False
        self.telegram_bot: Optional[Bot] = None
        
        # Validate configs in a background thread to avoid blocking startup
        threading.Thread(target=self._validate_notification_configs, daemon=True).start()
        
        self.db_queue: Queue = Queue()
        self.db_worker_running = True
        self.db_worker_thread = threading.Thread(target=self._db_writer_worker, daemon=True)
        self.db_worker_thread.start()

    def _validate_notification_configs(self) -> None:
        """Validates and tests both Telegram and Email configurations."""
        if TELEGRAM_AVAILABLE and self.config.telegram_enabled and self.config.telegram_bot_token and self.config.telegram_chat_ids:
            logger.info("Testing Telegram bot configuration...")
            try:
                # Use a fresh event loop for the background thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                bot = Bot(token=self.config.telegram_bot_token)
                loop.run_until_complete(bot.get_me())
                self.telegram_enabled = True
                self.telegram_bot = bot
                logger.info("Telegram bot validated successfully.")
            except Exception as e:
                logger.error(f"Failed to validate Telegram bot: {e}")
        
        self.email_enabled = False
        if self.config.sender_email and self.config.email_password and self.config.recipient_list:
            logger.info("Testing email configuration...")
            try:
                yagmail.SMTP(user=self.config.sender_email, password=self.config.email_password).close()
                self.email_enabled = True
                logger.info(f"Email configuration validated.")
            except Exception as e:
                logger.error(f"Failed to validate email configuration: {e}")

    def is_curfew_hours(self) -> bool:
        """Checks if the current time is within the configured curfew hours."""
        now = datetime.now().time()
        start, end = self.config.curfew_start, self.config.curfew_end
        return (start <= now <= end) if start <= end else (now >= start or now <= end)

    def handle_alert_event(self, event: Dict) -> None:
        """Public entry point to handle a new detection event."""
        if not self.detection_enabled: return

        viewport_id = event['viewport_id']

        # Signal the frontend on every detection so the highlight stays active
        self.socketio.emit('detection_in_progress', {'row': viewport_id[0], 'col': viewport_id[1]}, namespace='/')

        with self.alert_lock:
            if viewport_id in self.batch_timers and self.batch_timers[viewport_id].is_alive():
                # Capture additional screenshots for the batch (up to 5 total)
                batch_screenshots = sum(1 for e in self.pending_alerts[viewport_id] if e.get('screenshot_path'))
                if batch_screenshots < 5 and event.get('frame') is not None:
                    timestamp = datetime.fromtimestamp(event['timestamp'])
                    event['screenshot_path'] = self._take_screenshot(event['frame'], timestamp, event['viewport_name'])
                self.pending_alerts[viewport_id].append(event)
                if self.video_recorder.is_recording(viewport_id):
                    self.video_recorder.update_activity(viewport_id)
                return

            if time.time() - self.last_alert_times[viewport_id] < self.config.cooldown_period:
                if self.video_recorder.is_recording(viewport_id):
                    self.video_recorder.update_activity(viewport_id)
                return

            self._initiate_new_event_batch(event)

    def _initiate_new_event_batch(self, event: Dict) -> None:
        """Starts the batching window for a new event."""
        viewport_id = event['viewport_id']
        timestamp = datetime.fromtimestamp(event['timestamp'])
        event['screenshot_path'] = self._take_screenshot(event['frame'], timestamp, event['viewport_name'])

        self.pending_alerts[viewport_id].append(event)
        timer = threading.Timer(self.config.batch_window, self._process_alert_batch, args=[viewport_id])
        timer.daemon = True
        timer.start()
        self.batch_timers[viewport_id] = timer

    def _take_screenshot(self, frame: np.ndarray, timestamp: datetime, name: str) -> Optional[str]:
        """Saves a screenshot scaled to 720p max height for fast mobile delivery."""
        try:
            h, w = frame.shape[:2]
            if h > 720:
                scale = 720 / h
                frame = cv2.resize(frame, (int(w * scale), 720), interpolation=cv2.INTER_AREA)
            path = os.path.join("event_captures", f"{timestamp.strftime('%Y%m%d_%H%M%S')}_{name.replace(' ', '_')}.jpg")
            if cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 80]):
                logger.info(f"Screenshot saved: {path}")
                return path
            logger.error(f"Failed to save screenshot: {path}")
        except Exception as e:
            logger.error(f"Error taking screenshot for {name}: {e}")
        return None

    def _log_event_to_db(self, timestamp: datetime, name: str, conf: int, screen_path: Optional[str], vid_path: Optional[str]) -> None:
        """Queues event metadata for asynchronous database insertion."""
        self.db_queue.put((timestamp, name, conf, screen_path, vid_path))

    def _db_writer_worker(self) -> None:
        """Background worker that batches database writes."""
        while self.db_worker_running:
            batch = []
            try:
                while not self.db_queue.empty():
                    batch.append(self.db_queue.get_nowait())
            except Empty:
                pass
            
            if batch:
                try:
                    with sqlite3.connect(DB_FILE, check_same_thread=False) as conn:
                        cursor = conn.cursor()
                        cursor.executemany("INSERT INTO events (timestamp, viewport_name, confidence, screenshot_path, video_path) VALUES (?, ?, ?, ?, ?)",
                                         [(ts.isoformat(), n, c, s, v) for ts, n, c, s, v in batch])
                        conn.commit()
                    logger.info(f"Logged {len(batch)} events to database.")
                except Exception as e:
                    logger.error(f"Database batch write error: {e}")
            time.sleep(1)

    def _process_alert_batch(self, viewport_id: tuple) -> None:
        """Processes a batch of alerts, logs to DB, emits to frontend, and dispatches notifications."""
        with self.alert_lock:
            if not (alerts := self.pending_alerts.pop(viewport_id, [])): return
            self.last_alert_times[viewport_id] = time.time()
            self.batch_timers.pop(viewport_id, None)

        primary_alert = max(alerts, key=lambda x: x['confidence'])
        timestamp = datetime.fromtimestamp(primary_alert['timestamp'])
        
        self._log_event_to_db(timestamp, primary_alert['viewport_name'], primary_alert['confidence'], 
                              primary_alert.get('screenshot_path'), primary_alert.get('video_path'))

        self.socketio.emit('new_alert', {
            'viewport_name': primary_alert['viewport_name'], 'timestamp': timestamp.isoformat(),
            'confidence': primary_alert['confidence'], 'row': viewport_id[0], 'col': viewport_id[1],
            'video_path': primary_alert.get('video_path')
        }, namespace='/')

        if self.is_curfew_hours():
            # Move notification dispatch to a background thread to prevent blocking
            threading.Thread(target=self._dispatch_notifications, args=(primary_alert, alerts), daemon=True).start()

    def _dispatch_notifications(self, primary_alert: Dict, all_alerts: List[Dict]) -> None:
        """Handles the logic for sending email and/or Telegram notifications."""
        video_path = primary_alert.get('video_path')
        screenshot_paths = [a['screenshot_path'] for a in all_alerts if a.get('screenshot_path')]

        telegram_sent = False
        if self.telegram_enabled:
            message = self._prepare_telegram_message(all_alerts)
            telegram_sent = self._send_telegram_alert(message, screenshot_paths)

        if self.email_enabled and (not telegram_sent and self.config.email_as_fallback):
            subject = f"Security Alert - {primary_alert['viewport_name']}"
            content, attachments = self._prepare_email_content(all_alerts, screenshot_paths[0] if screenshot_paths else None, video_path)
            self._send_email(subject, content, attachments)

    def _prepare_email_content(self, alerts: List[Dict], screen_path: Optional[str], vid_path: Optional[str]) -> Tuple[str, List[str]]:
        # Stylized HTML content placeholder
        return "Email Content", [screen_path] if screen_path else []

    def _send_email(self, subject: str, content: str, attachments: Optional[List[str]] = None) -> None:
        """Sends an email alert in a background thread."""
        try:
            logger.info(f"Sending email alert to {self.config.recipient_list}")
            yag = yagmail.SMTP(user=self.config.sender_email, password=self.config.email_password)
            yag.send(to=self.config.recipient_list, subject=subject, contents=[content], attachments=attachments)
            logger.info("Email sent successfully.")
        except Exception as e:
            logger.error(f"Failed to send email: {e}")

    def _extract_video_clip(self, video_path: str, max_size_mb: int = 40) -> Optional[str]:
        """Extracts a smaller, web-compatible clip from a video file using ffmpeg."""
        if not video_path or not os.path.exists(video_path):
            logger.warning(f"Video file not found for extraction: {video_path}")
            return None

        # Wait briefly to ensure the recorder has finished writing the file
        time.sleep(5.0)

        try:
            output_path = video_path.replace(".mp4", "_clip.mp4")
            
            # Use ffmpeg to extract a 10-second clip and re-encode to H.264
            # Detections usually peak in the first few seconds, so we take the start.
            cmd = [
                'ffmpeg', '-y', 
                '-i', video_path, 
                '-t', '10', 
                '-c:v', 'libx264', 
                '-preset', 'ultrafast', 
                '-crf', '28', # Quality tradeoff for speed/size
                '-pix_fmt', 'yuv420p', 
                output_path
            ]
            
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info(f"Successfully extracted video clip: {output_path}")
            return output_path
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg extraction failed: {e.stderr}")
            return None
        except Exception as e:
            logger.error(f"Error extracting video clip: {e}")
            return None

    def _prepare_telegram_message(self, alerts: List[Dict]) -> str:
        """Formats the text message for a Telegram alert."""
        primary = max(alerts, key=lambda x: x['confidence'])
        ts = datetime.fromtimestamp(primary['timestamp'])
        return f"🚨 *SECURITY ALERT*\n\n*Camera:* {primary['viewport_name']}\n*Time:* {ts.strftime('%Y-%m-%d %H:%M:%S')}\n*Confidence:* {primary['confidence']}%"

    def _send_telegram_alert(self, message: str, screenshot_paths: Optional[List[str]] = None) -> bool:
        """Sends a Telegram alert with optional screenshot group."""
        if not self.telegram_enabled or not self.telegram_bot: return False

        try:
            logger.info(f"Sending Telegram alert with {len(screenshot_paths or [])} screenshot(s)...")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._async_send_telegram(message, screenshot_paths or []))
            return True
        except Exception as e:
            logger.error(f"Telegram alert failed: {e}")
            return False

    async def _async_send_telegram(self, message: str, screenshot_paths: List[str]) -> None:
        """Sends the Telegram message with a media group of screenshots."""
        if not self.telegram_bot: return

        # Filter to paths that actually exist
        valid_paths = [p for p in screenshot_paths if p and os.path.exists(p)]

        for chat_id in self.config.telegram_chat_ids:
            try:
                if len(valid_paths) > 1:
                    from telegram import InputMediaPhoto
                    media = []
                    for i, path in enumerate(valid_paths):
                        with open(path, 'rb') as f:
                            photo_data = f.read()
                        media.append(InputMediaPhoto(
                            media=photo_data,
                            caption=message if i == 0 else None,
                            parse_mode='Markdown' if i == 0 else None
                        ))
                    await self.telegram_bot.send_media_group(chat_id=chat_id, media=media)
                elif len(valid_paths) == 1:
                    with open(valid_paths[0], 'rb') as f:
                        await self.telegram_bot.send_photo(chat_id=chat_id, photo=f, caption=message, parse_mode='Markdown')
                else:
                    await self.telegram_bot.send_message(chat_id=chat_id, text=message, parse_mode='Markdown')
                logger.info(f"Telegram alert sent to chat {chat_id}.")
            except Exception as e:
                logger.error(f"Failed to send to chat {chat_id}: {e}")

    def stop(self) -> None:
        """Stops the alert manager and flushes pending database writes."""
        with self.alert_lock:
            for timer in self.batch_timers.values():
                if timer: timer.cancel()
        self.db_worker_running = False
        if self.db_worker_thread.is_alive():
            self.db_worker_thread.join(timeout=2.0)

class Viewport:
    """Represents a single logical area of a camera feed for processing."""
    def __init__(self, viewport_id: Tuple[int, int], config: Dict):
        self.id = viewport_id
        self.config = ViewportConfig(**config)
        self.motion_detector = MotionDetector(MotionDetectorConfig(self.config.sensitivity, self.config.motion_aggressiveness), self.id)
        self.frame_counter = 0
        self.last_detection_timestamp = 0.0
        self.detection_timeout_duration = 5.0

    def process_frame(self, frame: np.ndarray, yolo_manager: YOLOProcessManager) -> None:
        """Process frame for motion and submit to YOLO if needed."""
        self.frame_counter += 1
        motion_detected = self.motion_detector.check(frame)
        now = time.time()
        is_tracked = (now - self.last_detection_timestamp) < self.detection_timeout_duration
        if (motion_detected or is_tracked) and (self.frame_counter % self.config.yolo_interval == 0):
            try:
                yolo_manager.submit_detection_task(self.id, frame)
            except Full:
                logger.warning(f"YOLO detection queue full for viewport {self.id}. Frame dropped.")

    def update_config(self, new_config_data: Dict) -> None:
        """Updates the viewport's configuration dynamically."""
        as_dict = asdict(self.config)
        as_dict.update(new_config_data)
        self.config = ViewportConfig(**as_dict)
        self.motion_detector = MotionDetector(MotionDetectorConfig(self.config.sensitivity, self.config.motion_aggressiveness), self.id)
        logger.info(f"Configuration updated for viewport {self.id}")

class SecuritySystem:
    """The central orchestrator for the Argus security monitoring system."""
    def __init__(self, config_data: Dict, socketio: SocketIO, flask_app: Optional[Any] = None) -> None:
        self.config = config_data
        self.socketio = socketio
        self.flask_app = flask_app
        self.running = False
        enabled_cameras = [c for c in self.config['cameras'] if c.get('enabled', True)]
        self.grid_size = calculate_optimal_grid(len(enabled_cameras))
        self.viewport_configs = create_viewport_configs_from_cameras(self.config['cameras'], self.config['viewport_defaults'])
        
        self._initialize_managers()
        self._initialize_buffers()
        self._initialize_viewports()
        self._initialize_recorders_and_alerters()
        self._setup_frame_broker_subscriptions()

        self.worker_threads: List[threading.Thread] = []
        self._video_dimensions_updated = False
        self.encoded_frames: Dict[tuple, bytes] = {}
        self.encoding_lock = threading.Lock()

    def _initialize_managers(self) -> None:
        """Initializes core manager components."""
        mem_limit = self.config.get('system', {}).get('memory_limit_mb', 1024)
        self.memory_manager = GlobalMemoryManager(mem_limit)
        self.frame_pool = SharedFramePool(max_memory_mb=512)
        self.frame_broker = FrameBroker(max_memory_mb=min(512, mem_limit // 4), memory_manager=self.memory_manager)
        self.metrics = get_global_metrics()
        self.camera_manager = CameraManager(create_camera_configs(self.config), self.frame_broker)
        self.yolo_manager = YOLOProcessManager(
            model_path=self.config['system']['model_path'], 
            device=self.config['system'].get('yolo_device'), 
            max_queue_size=20
        )
        self.alert_queue: Queue = Queue(maxsize=100)

    def _initialize_buffers(self) -> None:
        """Initializes memory-bounded buffers for each viewport."""
        self.viewport_buffers, self.recording_buffers, self.display_buffers = {}, {}, {}
        vp_mem = 30; rec_mem = 50; disp_mem = 7
        
        for vp_id_str in self.viewport_configs.keys():
            r, c = map(int, vp_id_str.split(','))
            vp_id = (r, c)
            
            # Viewport (Processing) Buffer
            v_buf = MemoryBoundedBuffer(max_memory_bytes=vp_mem * 1024*1024, eviction_policy=EvictionPolicy.PRIORITY, name=f"viewport_{r}_{c}")
            self.viewport_buffers[vp_id] = v_buf
            self.memory_manager.register_buffer(v_buf)
            
            # Recording Buffer
            r_buf = MemoryBoundedBuffer(max_memory_bytes=rec_mem * 1024*1024, eviction_policy=EvictionPolicy.FIFO, name=f"recording_{r}_{c}")
            self.recording_buffers[vp_id] = r_buf
            self.memory_manager.register_buffer(r_buf)
            
            # Display Buffer
            d_buf = MemoryBoundedBuffer(max_memory_bytes=disp_mem * 1024*1024, eviction_policy=EvictionPolicy.LRU, name=f"display_{r}_{c}")
            self.display_buffers[vp_id] = d_buf
            self.memory_manager.register_buffer(d_buf)

    def _initialize_viewports(self) -> None:
        """Initializes Viewport instances from configuration."""
        self.viewports = {}
        for vp_id_str, vp_config in self.viewport_configs.items():
            r, c = map(int, vp_id_str.split(','))
            self.viewports[(r, c)] = Viewport((r, c), vp_config)

    def _initialize_recorders_and_alerters(self) -> None:
        """Initializes the video recorder and alert manager."""
        shared_lock = threading.RLock()
        cam_configs = create_camera_configs(self.config)
        self.recording_fps = cam_configs[0].fps if cam_configs else self.config['system'].get('max_fps', 25)
        
        rec_conf = self.config['video_recording']
        w, h = (1280 // self.grid_size[1]), (720 // self.grid_size[0]) # Default fallback size
        self.video_recorder = VideoClipRecorder(rec_conf, w, h, self.recording_fps, shared_lock)
        
        am_conf = self.config['alert_manager']
        alert_config_data = {
            'curfew_start': datetime.strptime(am_conf['curfew_start'], '%H:%M').time(),
            'curfew_end': datetime.strptime(am_conf['curfew_end'], '%H:%M').time(),
            'cooldown_period': am_conf['cooldown_period_seconds'],
            'batch_window': am_conf['batch_window_seconds'],
            'recipient_list': am_conf['recipient_list'],
            'sender_email': am_conf.get('sender_email'),
            'sender_name': am_conf.get('sender_name'),
            'email_password': am_conf.get('email_password'),
            'telegram_bot_token': am_conf.get('telegram_bot_token'),
            'telegram_chat_ids': am_conf.get('telegram_chat_ids'),
            'telegram_enabled': am_conf.get('telegram_enabled', False),
            'email_as_fallback': am_conf.get('email_as_fallback', True)
        }
        self.alert_manager = AlertManager(AlertConfig(**alert_config_data), self.socketio, shared_lock, self.video_recorder, self.flask_app)

    def _setup_frame_broker_subscriptions(self) -> None:
        """Subscribes the main frame processing callback to each camera."""
        for cam_config in create_camera_configs(self.config):
            if cam_config.enabled:
                self.frame_broker.register_camera(cam_config.camera_name, base_fps=cam_config.fps)
                self.frame_broker.subscribe_to_camera(cam_config.camera_name, self._process_camera_frame)

    def start(self) -> bool:
        """Starts the security system and all its components."""
        self.running = True
        if not self.camera_manager.start_all():
            logger.warning("No cameras started successfully.")
        if not self.yolo_manager.start_process():
            logger.error("Failed to start YOLO process. Exiting.")
            self.camera_manager.stop_all()
            return False
        
        self._start_worker_threads()
        logger.info("SecuritySystem processing pipeline started.")
        return True

    def _start_worker_threads(self) -> None:
        """Initializes and starts all background worker threads."""
        self.worker_threads = [
            threading.Thread(target=self._metrics_worker, daemon=True),
            threading.Thread(target=self._yolo_result_processor_worker, daemon=True),
            threading.Thread(target=self._event_handler_worker, daemon=True),
            threading.Thread(target=self._jpeg_encoding_worker, daemon=True)
        ]
        
        num_vps = len(self.viewports)
        num_workers = max(1, min(num_vps // 2, os.cpu_count() // 2))
        for i in range(num_workers):
            self.worker_threads.append(threading.Thread(target=self._viewport_worker, daemon=True, name=f"ViewportWorker-{i+1}"))
        
        for thread in self.worker_threads:
            thread.start()

    def stop(self) -> None:
        """Stops the security system and cleans up all components and threads."""
        self.running = False
        logger.info("Stopping SecuritySystem components...")
        self.camera_manager.stop_all()
        self.yolo_manager.stop_process()
        self.frame_broker.shutdown()
        self.alert_manager.stop()
        self.metrics.shutdown()
        self.frame_pool.shutdown()
        for thread in self.worker_threads:
            thread.join(timeout=2.0)
        logger.info("SecuritySystem stopped.")

    def _process_camera_frame(self, camera_name: str, frame: np.ndarray, timestamp: float) -> None:
        """Callback to process a frame from a camera."""
        if timestamp % 5.0 < 0.04:
            record_metric("camera_fps", 1.0, {"camera": camera_name})
        
        for vp_id, viewport in self.viewports.items():
            if viewport.config.camera_name == camera_name:
                self._handle_viewport_frame(vp_id, frame, timestamp)

    def _handle_viewport_frame(self, vp_id: tuple, frame: np.ndarray, timestamp: float) -> None:
        """Processes a frame for a specific viewport."""
        if not self._video_dimensions_updated:
            h, w, _ = frame.shape
            self.video_recorder.update_dimensions(w, h)
            self._video_dimensions_updated = True

        self.display_buffers[vp_id].add_frame(frame, timestamp, 0)
        
        scale = self.viewports[vp_id].config.scale_factor
        proc_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST) if scale < 1.0 else frame
        
        if frame_ref := self.frame_pool.add_frame(proc_frame, timestamp):
            priority = 1 if time.time() - self.viewports[vp_id].last_detection_timestamp < 10 else 0
            
            if viewport_ref := self.frame_pool.acquire_reference(frame_ref.frame_id):
                if not self.viewport_buffers[vp_id].add_frame(viewport_ref, timestamp, priority):
                    viewport_ref.release()
            
            rec_frame = cv2.resize(frame, (self.video_recorder.frame_width, self.video_recorder.frame_height), interpolation=cv2.INTER_NEAREST)
            self.recording_buffers[vp_id].add_frame(rec_frame, timestamp, priority)
            
            frame_ref.release()

        if self.video_recorder.is_recording(vp_id):
            try:
                self.video_recorder.recording_state[vp_id]['frame_queue'].put_nowait(frame)
            except Full:
                pass

    def _metrics_worker(self) -> None:
        """Worker thread to collect and update system metrics periodically."""
        while self.running:
            try:
                stats = self.memory_manager.get_global_stats()
                record_metric("memory_usage_mb", stats['total_memory_usage_mb'])
                
                if self.memory_manager.is_memory_pressure_critical():
                    logger.warning("Critical memory pressure detected. Attempting cleanup.")
                    self.memory_manager.handle_memory_pressure()
            except Exception as e:
                logger.error(f"Metrics worker error: {e}")
            time.sleep(5.0)

    def _viewport_worker(self) -> None:
        """Worker thread for processing frames from assigned viewports."""
        worker_name = threading.current_thread().name
        try:
            worker_num = int(worker_name.split('-')[1])
        except (IndexError, ValueError):
            worker_num = 1
            
        total_workers = sum(1 for t in self.worker_threads if 'ViewportWorker' in str(t.name))
        if total_workers == 0: total_workers = 1
        
        assigned_vps = [vp for i, vp in enumerate(sorted(self.viewports.keys())) if i % total_workers == (worker_num - 1)]
        logger.info(f"{worker_name} assigned to viewports: {assigned_vps}")

        while self.running:
            for vp_id in assigned_vps:
                if (entry := self.viewport_buffers[vp_id].wait_and_pop_entry(timeout=0.05)) and (frame_data := entry.get_frame_data()) is not None:
                    self.viewports[vp_id].process_frame(frame_data, self.yolo_manager)
                    entry.release()

    def _yolo_result_processor_worker(self) -> None:
        """Worker thread to process detection results from the YOLO subprocess."""
        while self.running:
            try:
                for vp_id, detections, _ in self.yolo_manager.get_detection_results(max_results=10):
                    if viewport := self.viewports.get(vp_id):
                        confident_detections = [d for d in detections if d['confidence'] >= viewport.config.min_confidence]
                        if confident_detections:
                            viewport.last_detection_timestamp = time.time()
                            primary = max(confident_detections, key=lambda p: p['confidence'])
                            event = {
                                'viewport_id': vp_id, 
                                'viewport_name': viewport.config.name, 
                                'timestamp': time.time(),
                                'confidence': int(primary['confidence'] * 100), 
                                'frame': None
                            }
                            try:
                                self.alert_queue.put_nowait(event)
                            except Full:
                                logger.warning(f"Alert queue full, dropping event for {vp_id}")
            except Exception as e:
                logger.error(f"YOLO result processor error: {e}")
            time.sleep(0.01)

    def _event_handler_worker(self) -> None:
        """Worker thread to handle events from the alert queue."""
        while self.running:
            try:
                event = self.alert_queue.get(timeout=1.0)
                if not self.alert_manager.detection_enabled: continue
                
                vp_id = event['viewport_id']
                if event.get('frame') is None:
                    event['frame'] = self.display_buffers[vp_id].get_latest_frame()

                if event['frame'] is not None:
                    event['video_path'] = self.video_recorder.handle_alert_event(event, self.recording_buffers[vp_id])
                    self.alert_manager.handle_alert_event(event)
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Event handler worker error: {e}")

    def _jpeg_encoding_worker(self) -> None:
        """Worker thread to pre-encode frames to JPEG for web streaming."""
        target_fps = self.config.get('system', {}).get('dashboard_fps', 12)
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 55]
        while self.running:
            try:
                start_time = time.time()
                for vp_id, buffer in self.display_buffers.items():
                    if (frame := buffer.get_latest_frame()) is not None:
                        _, enc_buffer = cv2.imencode('.jpg', frame, encode_params)
                        with self.encoding_lock:
                            self.encoded_frames[vp_id] = enc_buffer.tobytes()

                elapsed = time.time() - start_time
                time.sleep(max(0, (1.0 / target_fps) - elapsed))
            except Exception as e:
                logger.error(f"JPEG encoding worker error: {e}")
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
