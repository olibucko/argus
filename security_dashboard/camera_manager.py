import cv2
import time
import threading
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

@dataclass
class CameraConfig:
    """Configuration for a single camera."""
    camera_id: str  # Device index or IP URL
    camera_name: str = "Unknown Camera"
    width: int = 1280
    height: int = 720
    fps: int = 25
    enabled: bool = True
    retry_interval: float = 5.0  # Seconds between reconnection attempts

class CameraCapture:
    """Manages a single camera capture with auto-reconnection."""
    
    def __init__(self, config: CameraConfig):
        self.config = config
        self.cap: Optional[cv2.VideoCapture] = None
        self.running = False
        self.connected = False
        self.last_frame_time = 0
        self.frame_count = 0
        self.last_read_time = 0  # Track when we last read a frame for FPS throttling
        
    def connect(self) -> bool:
        """Attempt to connect to the camera."""
        try:
            if isinstance(self.config.camera_id, int):
                self.cap = cv2.VideoCapture(self.config.camera_id, cv2.CAP_ANY)
            else:
                self.cap = cv2.VideoCapture(self.config.camera_id)

            if not self.cap.isOpened():
                print(f"[ERROR] Failed to open video source: {self.config.camera_id}")
                return False

            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.config.fps)

            # Minimize buffer size for RTSP to reduce latency and prevent frame bursts
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Test frame capture
            ret, _ = self.cap.read()
            if not ret:
                self.cap.release()
                return False
                
            self.connected = True
            print(f"[OK] Camera '{self.config.camera_name}' connected successfully")
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to connect camera '{self.config.camera_name}': {e}")
            if self.cap:
                self.cap.release()
            return False
    
    def disconnect(self):
        """Disconnect from the camera."""
        self.connected = False
        if self.cap:
            self.cap.release()
            self.cap = None
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a frame from the camera."""
        if not self.connected or not self.cap:
            return False, None

        try:
            ret, frame = self.cap.read()
            if ret:
                self.last_frame_time = time.time()
                self.frame_count += 1
                return True, frame
            else:
                self.connected = False
                return False, None
        except Exception as e:
            print(f"Error reading from camera '{self.config.camera_name}': {e}")
            self.connected = False
            return False, None

class CameraManager:
    """Manages multiple cameras and their capture threads."""

    def __init__(self, camera_configs: List[CameraConfig], frame_broker=None):
        self.cameras: Dict[str, CameraCapture] = {}
        self.capture_threads: Dict[str, threading.Thread] = {}
        self.frame_broker = frame_broker  # Can be None for backward compatibility
        self.running = False
        
        # Initialize cameras
        for config in camera_configs:
            if config.enabled:
                camera_name = config.camera_name
                self.cameras[camera_name] = CameraCapture(config)
                
    def start_all(self) -> bool:
        """Start all camera capture threads."""
        self.running = True
        success_count = 0
        
        for camera_name, camera in self.cameras.items():
            if camera.connect():
                thread = threading.Thread(
                    target=self._camera_loop, 
                    args=(camera_name, camera),
                    daemon=True
                )
                thread.start()
                self.capture_threads[camera_name] = thread
                success_count += 1
            else:
                print(f"[WARNING] Failed to start camera: {camera_name}")
                
        print(f"[STARTUP] Started {success_count}/{len(self.cameras)} cameras")
        return success_count > 0
    
    def stop_all(self):
        """Stop all camera capture threads."""
        self.running = False
        
        # Disconnect all cameras
        for camera in self.cameras.values():
            camera.disconnect()
            
        # Wait for threads to finish
        for thread in self.capture_threads.values():
            thread.join(timeout=2.0)
            
        print("[CAMERA] All cameras stopped")
    
    def _camera_loop(self, camera_name: str, camera: CameraCapture):
        """Main loop for a single camera capture thread."""
        consecutive_failures = 0
        max_failures = 10

        # Calculate target frame interval based on camera FPS
        target_interval = 1.0 / camera.config.fps

        # Initialize timing
        next_frame_time = time.time()

        while self.running:
            if not camera.connected:
                # Attempt reconnection
                print(f"[CAMERA] Attempting to reconnect camera: {camera_name}")
                if camera.connect():
                    consecutive_failures = 0
                    next_frame_time = time.time()  # Reset timing
                else:
                    consecutive_failures += 1
                    if consecutive_failures >= max_failures:
                        print(f"[ERROR] Camera {camera_name} failed too many times. Stopping.")
                        break
                    time.sleep(camera.config.retry_interval)
                    continue

            # Wait until next frame time
            current_time = time.time()
            sleep_time = next_frame_time - current_time
            if sleep_time > 0.001:  # Only sleep if more than 1ms needed
                time.sleep(sleep_time)

            # Schedule next frame (regardless of processing time)
            next_frame_time += target_interval

            # Prevent drift - if we fall behind, reset to current time
            if next_frame_time < current_time - target_interval:
                next_frame_time = current_time

            # Flush RTSP buffer to get the latest frame (prevents reading stale buffered frames)
            # For RTSP streams, read and discard to ensure fresh frame
            if isinstance(camera.config.camera_id, str) and camera.config.camera_id.startswith('rtsp'):
                # Grab (don't decode) to flush buffer quickly
                for _ in range(2):  # Flush 2 buffered frames for ~100ms freshness
                    if camera.cap:
                        camera.cap.grab()

            # Read frame (now should be the latest)
            ret, frame = camera.read_frame()
            if ret and frame is not None:
                consecutive_failures = 0

                # Submit frame to frame broker
                timestamp = time.time()

                if self.frame_broker:
                    # Use new frame broker system
                    if not self.frame_broker.submit_frame(camera_name, frame, timestamp):
                        # Get detailed camera status to understand why frame was dropped
                        camera_status = self.frame_broker.get_camera_status()
                        if camera_name in camera_status:
                            status = camera_status[camera_name]
                            print(f"[WARNING] Frame dropped from {camera_name} - drop_rate: {status['drop_rate']*100:.1f}%, "
                                  f"consecutive_drops: {status['consecutive_drops']}, current_fps: {status['current_fps']:.1f}")
                        else:
                            print(f"[ERROR] Frame dropped from {camera_name} - camera not registered with frame broker!")

            else:
                consecutive_failures += 1
                print(f"[WARNING] Frame read failed for {camera_name} (failures: {consecutive_failures})")

                if consecutive_failures >= max_failures:
                    print(f"[ERROR] Too many failures for {camera_name}. Will attempt reconnection.")
                    camera.disconnect()
                    time.sleep(camera.config.retry_interval)
                    next_frame_time = time.time()  # Reset timing after reconnect
    
    def get_camera_status(self) -> Dict[str, Dict]:
        """Get status of all cameras."""
        status = {}
        for name, camera in self.cameras.items():
            status[name] = {
                'connected': camera.connected,
                'frame_count': camera.frame_count,
                'last_frame_time': camera.last_frame_time,
                'config': camera.config
            }
        return status
    
    def restart_camera(self, camera_name: str) -> bool:
        """Restart a specific camera."""
        if camera_name not in self.cameras:
            return False
            
        camera = self.cameras[camera_name]
        camera.disconnect()
        time.sleep(1.0)
        return camera.connect()

# Helper function to create camera configs from config file
def create_camera_configs(config_data: dict) -> List[CameraConfig]:
    """Create camera config objects from configuration data."""
    camera_configs = []
    
    for camera_data in config_data.get('cameras', []):
        config = CameraConfig(
            camera_name=camera_data['camera_name'],
            camera_id=camera_data['camera_id'],
            width=camera_data.get('width', 1280),
            height=camera_data.get('height', 720),
            fps=camera_data.get('fps', 30),
            enabled=camera_data.get('enabled', True),
            retry_interval=camera_data.get('retry_interval', 5.0)
        )
        camera_configs.append(config)
        
    return camera_configs