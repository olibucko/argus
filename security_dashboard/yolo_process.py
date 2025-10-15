# yolo_process.py - Separate YOLO detection process
import multiprocessing as mp
from multiprocessing import shared_memory
import time
import numpy as np
import cv2
import torch
from queue import Empty
import logging
from typing import Dict, List, Any, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - Console - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class YOLODetectionProcess:
    """
    Standalone YOLO detection process that runs independently of the main application.
    Communicates via multiprocessing queues for maximum performance.
    """
    
    def __init__(self, model_path: str, device: str = None, inference_size: int = 640):
        self.model_path = model_path
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.inference_size = inference_size
        self.model = None
        
    def initialize_model(self):
        """Initialize the YOLO model in the subprocess context."""
        try:
            import warnings
            warnings.filterwarnings("ignore", category=FutureWarning)
            warnings.filterwarnings("ignore", message=".*autocast.*")
            logger.info(f"Loading YOLO model on device: {self.device}")
            
            # Load model in subprocess
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', 
                                      path=self.model_path, _verbose=False)
            self.model.to(self.device)
            self.model.float()
            
            # Configure model settings
            self.model.classes = [0]  # Only detect 'person' class
            self.model.conf = 0.3  # Minimum confidence (viewports will filter further)
            
            # GPU warm-up if using CUDA
            if self.device != 'cpu':
                logger.info("Warming up GPU inference...")
                dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
                _ = self.model(dummy_img, size=self.inference_size)
                logger.info("GPU warm-up complete")
                
            logger.info("YOLO model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            return False
    
    def detect_objects(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Perform object detection on a single frame."""
        if self.model is None or frame is None or frame.size == 0:
            return []
            
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run inference
            results = self.model(rgb_frame, size=self.inference_size)
            
            # Process results
            predictions = results.pred[0].cpu().numpy()
            person_predictions = [p for p in predictions if p[5] == 0]  # class 0 is 'person'
            
            detections = []
            for pred in person_predictions:
                detections.append({
                    "box": [int(val) for val in pred[:4]],
                    "confidence": float(pred[4]),
                    "class_id": int(pred[5])
                })
                
            return detections
            
        except Exception as e:
            logger.error(f"Error during YOLO inference: {e}")
            return []

def yolo_worker_process(input_queue: mp.Queue, output_queue: mp.Queue,
                       model_path: str, device: str = None):
    """
    Main worker function that runs in the separate YOLO process.

    Args:
        input_queue: Queue containing (task_id, viewport_id, shm_name, shape, dtype) tuples
        output_queue: Queue for sending back (task_id, viewport_id, detections, inference_time) tuples
        model_path: Path to the YOLO model file
        device: Device to run inference on ('cuda', 'cpu', etc.)
    """

    # Initialize the detection process
    detector = YOLODetectionProcess(model_path, device)

    if not detector.initialize_model():
        logger.error("Failed to initialize YOLO model. Exiting process.")
        return

    logger.info("YOLO worker process started and ready for inference")

    # Track shared memory segments for cleanup
    active_shm = {}

    # Main processing loop
    while True:
        try:
            # Get task from queue with timeout
            task_data = input_queue.get(timeout=1.0)

            if task_data is None:  # Shutdown signal
                logger.info("Received shutdown signal. Exiting YOLO process.")
                break

            task_id, viewport_id, shm_name, shape, dtype = task_data

            # Access shared memory and reconstruct frame
            try:
                shm = shared_memory.SharedMemory(name=shm_name)
                frame = np.ndarray(shape, dtype=dtype, buffer=shm.buf)

                # Perform detection
                start_time = time.time()
                detections = detector.detect_objects(frame)
                inference_time = time.time() - start_time

                # Close shared memory (don't unlink - main process will do that)
                shm.close()

                # Send results back
                result = (task_id, viewport_id, detections, inference_time)
                output_queue.put(result)

                # Log performance metrics periodically
                if task_id % 50 == 0:
                    logger.info(f"Processed task {task_id} for viewport {viewport_id} "
                              f"in {inference_time:.3f}s, found {len(detections)} detections")

            except FileNotFoundError:
                logger.error(f"Shared memory '{shm_name}' not found for task {task_id}")
                continue
            except Exception as e:
                logger.error(f"Error accessing shared memory: {e}")
                continue

        except Empty:
            continue
        except Exception as e:
            logger.error(f"Error in YOLO worker process: {e}")
            continue

    # Cleanup any remaining shared memory references
    for shm in active_shm.values():
        try:
            shm.close()
        except:
            pass

    logger.info("YOLO worker process terminated")


class YOLOProcessManager:
    """
    Manages the YOLO detection subprocess and handles communication.
    Uses shared memory for efficient frame passing.
    """

    def __init__(self, model_path: str, device: str = None, max_queue_size: int = 20):
        self.model_path = model_path
        self.device = device
        self.max_queue_size = max_queue_size

        # Communication queues
        self.input_queue = mp.Queue(maxsize=max_queue_size)
        self.output_queue = mp.Queue(maxsize=max_queue_size * 2)

        # Process management
        self.process = None
        self.task_counter = 0
        self.pending_tasks = {}  # task_id -> (viewport_id, timestamp, shm_name)

        # Shared memory management
        self.active_shm = {}  # task_id -> SharedMemory object

        # Performance tracking
        self.total_inferences = 0
        self.total_inference_time = 0.0
        self.total_serialization_time = 0.0  # Track time saved
        
    def start_process(self):
        """Start the YOLO detection subprocess."""
        try:
            self.process = mp.Process(
                target=yolo_worker_process,
                args=(self.input_queue, self.output_queue, self.model_path, self.device),
                daemon=True
            )
            self.process.start()
            logger.info(f"Started YOLO process with PID: {self.process.pid}")
            
            # Wait a moment for initialization
            time.sleep(2.0)
            
            if not self.process.is_alive():
                raise RuntimeError("YOLO process failed to start")
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to start YOLO process: {e}")
            return False
    
    def stop_process(self):
        """Stop the YOLO detection subprocess and cleanup shared memory."""
        if self.process and self.process.is_alive():
            # Send shutdown signal
            self.input_queue.put(None)

            # Wait for graceful shutdown
            self.process.join(timeout=5.0)

            if self.process.is_alive():
                logger.warning("YOLO process didn't shut down gracefully, terminating...")
                self.process.terminate()
                self.process.join(timeout=2.0)

        # Cleanup all shared memory segments
        self._cleanup_all_shared_memory()
        logger.info("YOLO process stopped and shared memory cleaned up")
    
    def submit_detection_task(self, viewport_id: tuple, frame: np.ndarray) -> Optional[int]:
        """
        Submit a frame for YOLO detection using shared memory.

        Args:
            viewport_id: Identifier for the viewport
            frame: Frame to process

        Returns:
            Task ID if submitted successfully, None if queue is full
        """
        try:
            if not self.process or not self.process.is_alive():
                return None

            task_id = self.task_counter
            self.task_counter += 1

            # Create shared memory for this frame
            shm_name = f"yolo_frame_{task_id}_{time.time_ns()}"
            try:
                shm = shared_memory.SharedMemory(create=True, size=frame.nbytes, name=shm_name)

                # Copy frame data to shared memory
                shm_array = np.ndarray(frame.shape, dtype=frame.dtype, buffer=shm.buf)
                shm_array[:] = frame[:]

                # Store shared memory reference for cleanup
                self.active_shm[task_id] = shm

                # Send metadata to worker process (much smaller than serialized frame)
                metadata = (task_id, viewport_id, shm_name, frame.shape, frame.dtype)
                self.input_queue.put_nowait(metadata)

                # Track pending task with shared memory name
                self.pending_tasks[task_id] = (viewport_id, time.time(), shm_name)

                return task_id

            except Exception as e:
                logger.error(f"Failed to create shared memory: {e}")
                # Cleanup shared memory if creation failed
                if task_id in self.active_shm:
                    self._cleanup_shared_memory(task_id)
                return None

        except Exception as e:
            logger.error(f"Failed to submit detection task: {e}")
            return None
    
    def get_detection_results(self, max_results: int = 10) -> List[tuple]:
        """
        Retrieve completed detection results and cleanup shared memory.

        Args:
            max_results: Maximum number of results to retrieve

        Returns:
            List of (viewport_id, detections, inference_time) tuples
        """
        results = []

        for _ in range(max_results):
            try:
                # Non-blocking get
                result = self.output_queue.get_nowait()
                task_id, viewport_id, detections, inference_time = result

                # Update performance metrics
                self.total_inferences += 1
                self.total_inference_time += inference_time

                # Cleanup shared memory for this task
                self._cleanup_shared_memory(task_id)

                # Clean up pending tasks
                self.pending_tasks.pop(task_id, None)

                results.append((viewport_id, detections, inference_time))

            except Empty:
                break
            except Exception as e:
                logger.error(f"Error retrieving detection results: {e}")
                break

        return results
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics for the YOLO process."""
        if self.total_inferences == 0:
            return {"avg_inference_time": 0.0, "total_inferences": 0}
            
        return {
            "avg_inference_time": self.total_inference_time / self.total_inferences,
            "total_inferences": self.total_inferences,
            "pending_tasks": len(self.pending_tasks)
        }
    
    def cleanup_stale_tasks(self, timeout: float = 30.0):
        """Remove tasks that have been pending too long and cleanup their shared memory."""
        current_time = time.time()
        stale_tasks = [
            task_id for task_id, (_, timestamp, _) in self.pending_tasks.items()
            if current_time - timestamp > timeout
        ]

        for task_id in stale_tasks:
            # Cleanup shared memory for stale task
            self._cleanup_shared_memory(task_id)
            self.pending_tasks.pop(task_id, None)

        if stale_tasks:
            logger.warning(f"Cleaned up {len(stale_tasks)} stale detection tasks")

    def _cleanup_shared_memory(self, task_id: int):
        """Cleanup shared memory for a specific task."""
        if task_id in self.active_shm:
            try:
                shm = self.active_shm.pop(task_id)
                shm.close()
                shm.unlink()
            except Exception as e:
                logger.debug(f"Error cleaning up shared memory for task {task_id}: {e}")

    def _cleanup_all_shared_memory(self):
        """Cleanup all active shared memory segments."""
        for task_id in list(self.active_shm.keys()):
            self._cleanup_shared_memory(task_id)
        logger.info(f"Cleaned up all shared memory segments")