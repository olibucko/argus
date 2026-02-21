"""
YOLO Process - Independent subprocess for high-performance object detection.

This module isolates the YOLOv5 inference engine into a separate process.
It uses shared memory for "zero-copy" frame transfer from the main process,
ensuring that the main application thread remains responsive and the UI is fluid.
"""

import multiprocessing as mp
from multiprocessing import shared_memory
import time
import numpy as np
import cv2
import torch
from queue import Empty
import logging
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class YOLODetectionProcess:
    """
    Handles the initialization and execution of the YOLOv5 model.

    Attributes:
        model_path (str): Path to the .pt weights file.
        device (str): Computation device ('cuda' or 'cpu').
        inference_size (int): Image size for the model (e.g., 640).
    """

    def __init__(self, model_path: str, device: Optional[str] = None, inference_size: int = 640) -> None:
        """
        Initializes the detection engine.

        Args:
            model_path (str): Path to weights.
            device (str, optional): Device override.
            inference_size (int): Input resolution.
        """
        self.model_path: str = model_path
        self.device: str = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.inference_size: int = inference_size
        self.model: Optional[Any] = None

    def initialize_model(self) -> bool:
        """
        Loads the YOLOv5 model into memory and performs a warm-up.

        Returns:
            bool: True if successful.
        """
        try:
            import warnings
            warnings.filterwarnings("ignore", category=FutureWarning)
            logger.info(f"Loading YOLO model on {self.device}...")
            
            # Load model from local path or Torch Hub
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.model_path, _verbose=False)
            self.model.to(self.device).float()
            
            # Configure engine
            self.model.classes = [0]  # Detect 'person' only
            self.model.conf = 0.3
            
            if self.device != 'cpu':
                logger.info("Warming up GPU...")
                self.model(np.zeros((640, 640, 3), dtype=np.uint8), size=self.inference_size)
                
            logger.info("YOLO model initialized.")
            return True
        except Exception as e:
            logger.error(f"YOLO initialization failed: {e}")
            return False

    def detect_objects(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Performs inference on a single frame.

        Args:
            frame (np.ndarray): The raw image data.

        Returns:
            List[Dict]: A list of detection results (box, confidence, etc.).
        """
        if self.model is None or frame is None:
            return []
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.model(rgb, size=self.inference_size)
            preds = results.pred[0].cpu().numpy()
            
            return [{
                "box": [int(val) for val in p[:4]],
                "confidence": float(p[4]),
                "class_id": int(p[5])
            } for p in preds if p[5] == 0]
        except Exception as e:
            logger.error(f"YOLO inference error: {e}")
            return []


def yolo_worker_process(input_queue: mp.Queue, output_queue: mp.Queue, model_path: str, device: Optional[str] = None) -> None:
    """
    Target function for the multiprocessing.Process.

    Args:
        input_queue (mp.Queue): Receives task metadata.
        output_queue (mp.Queue): Sends back results.
        model_path (str): Path to the model.
        device (str, optional): Target device.
    """
    detector = YOLODetectionProcess(model_path, device)
    if not detector.initialize_model(): return

    while True:
        try:
            task = input_queue.get(timeout=1.0)
            if task is None: break  # Shutdown signal

            task_id, viewport_id, shm_name, shape, dtype = task
            try:
                shm = shared_memory.SharedMemory(name=shm_name)
                frame = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
                
                start = time.time()
                detections = detector.detect_objects(frame)
                duration = time.time() - start
                
                shm.close()
                output_queue.put((task_id, viewport_id, detections, duration), timeout=0.5)
            except Exception as e:
                logger.error(f"Task {task_id} failed: {e}")
        except Empty:
            continue
    logger.info("YOLO worker process exiting.")


class YOLOProcessManager:
    """
    Manages the lifecycle and communication of the YOLO subprocess.

    Attributes:
        input_queue (mp.Queue): Queue for sending frame metadata to the worker.
        output_queue (mp.Queue): Queue for receiving results from the worker.
        active_shm (Dict): Map of task IDs to SharedMemory segments.
    """

    def __init__(self, model_path: str, device: Optional[str] = None, max_queue_size: int = 20) -> None:
        """
        Initializes the manager.

        Args:
            model_path (str): Weights file path.
            device (str, optional): Computation device.
            max_queue_size (int): Maximum pending tasks.
        """
        self.model_path: str = model_path
        self.device: Optional[str] = device
        self.input_queue: mp.Queue = mp.Queue(maxsize=max_queue_size)
        self.output_queue: mp.Queue = mp.Queue(maxsize=max_queue_size * 2)
        self.process: Optional[mp.Process] = None
        self.task_counter: int = 0
        self.active_shm: Dict[int, shared_memory.SharedMemory] = {}
        self.pending_tasks: Dict[int, Tuple[tuple, float, str]] = {}
        
        # Stats
        self.total_inferences: int = 0
        self.total_inference_time: float = 0.0

    def start_process(self) -> bool:
        """Starts the worker subprocess."""
        try:
            self.process = mp.Process(
                target=yolo_worker_process,
                args=(self.input_queue, self.output_queue, self.model_path, self.device),
                daemon=True
            )
            self.process.start()
            time.sleep(2.0)
            return self.process.is_alive()
        except Exception as e:
            logger.error(f"Failed to start YOLO process: {e}")
            return False

    def stop_process(self) -> None:
        """Terminates the worker process and cleans up memory."""
        if self.process and self.process.is_alive():
            self.input_queue.put(None)
            self.process.join(timeout=5.0)
            if self.process.is_alive(): self.process.terminate()
        self._cleanup_all_shared_memory()

    def submit_detection_task(self, viewport_id: tuple, frame: np.ndarray) -> Optional[int]:
        """
        Places a frame into shared memory and notifies the worker.

        Args:
            viewport_id (tuple): Source viewport ID.
            frame (np.ndarray): The image to process.

        Returns:
            Optional[int]: The task ID if successful.
        """
        if not self.process or not self.process.is_alive(): return None
        task_id = self.task_counter
        self.task_counter += 1
        shm_name = f"yolo_{task_id}_{time.time_ns()}"

        try:
            shm = shared_memory.SharedMemory(create=True, size=frame.nbytes, name=shm_name)
            np.ndarray(frame.shape, dtype=frame.dtype, buffer=shm.buf)[:] = frame[:]
            self.active_shm[task_id] = shm
            self.input_queue.put_nowait((task_id, viewport_id, shm_name, frame.shape, frame.dtype))
            self.pending_tasks[task_id] = (viewport_id, time.time(), shm_name)
            return task_id
        except Exception as e:
            logger.error(f"Task submission failed: {e}")
            if task_id in self.active_shm: self._cleanup_shared_memory(task_id)
            return None

    def get_detection_results(self, max_results: int = 10) -> List[tuple]:
        """Retrieves and cleans up finished tasks from the worker."""
        results = []
        for _ in range(max_results):
            try:
                task_id, viewport_id, detections, duration = self.output_queue.get_nowait()
                self.total_inferences += 1
                self.total_inference_time += duration
                self.pending_tasks.pop(task_id, None)
                self._cleanup_shared_memory(task_id)
                results.append((viewport_id, detections, duration))
            except Empty:
                break
        return results

    def get_performance_stats(self) -> Dict[str, float]:
        """Returns averages for inference time and queue state."""
        avg = self.total_inference_time / self.total_inferences if self.total_inferences > 0 else 0
        return {"avg_inference_time": avg, "total_inferences": self.total_inferences, "pending_tasks": len(self.pending_tasks)}

    def cleanup_stale_tasks(self, timeout: float = 30.0) -> None:
        """Finds and destroys shared memory for tasks that took too long."""
        now = time.time()
        stale = [tid for tid, (_, ts, _) in self.pending_tasks.items() if now - ts > timeout]
        for tid in stale:
            self._cleanup_shared_memory(tid)
            self.pending_tasks.pop(tid, None)
        if stale: logger.warning(f"Cleaned up {len(stale)} stale YOLO tasks.")

    def _cleanup_shared_memory(self, task_id: int) -> None:
        """Closes and unlinks a specific shared memory segment."""
        if shm := self.active_shm.pop(task_id, None):
            try:
                shm.close()
                shm.unlink()
            except Exception:
                pass

    def _cleanup_all_shared_memory(self) -> None:
        """Unlinks all remaining shared memory segments."""
        for tid in list(self.active_shm.keys()): self._cleanup_shared_memory(tid)
