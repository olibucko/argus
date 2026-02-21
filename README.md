# Argus — Real-Time Multi-Camera Surveillance System

Argus is a real-time security monitoring platform that combines multi-camera RTSP ingestion with AI-powered person detection, motion-gated inference, and automated alerting. Built for continuous, unattended operation in environments where reliable detection and fast notification matter — remote sites, industrial yards, and perimeter monitoring.

![Argus Dashboard Demo](demo.gif)

## Key Features

- **Multi-camera RTSP streaming** with per-camera threads, auto-reconnection, and FPS throttling
- **YOLOv5 person detection** isolated in a subprocess with shared-memory frame transfer for zero-copy inference
- **Motion-gated detection** using Lucas-Kanade optical flow to avoid running YOLO on static scenes
- **Automated alerting** via Telegram and email with configurable curfew hours, cooldown periods, and batch windowing
- **Pre/post-event video recording** capturing 5 seconds before and after each detection
- **Memory-bounded frame management** with FIFO, LRU, and priority-based eviction policies
- **Real-time web dashboard** with live camera feeds via Flask-SocketIO
- **System health metrics** tracking FPS, latency, memory pressure, and queue depths with threshold-based alerts

## Architecture

```
RTSP Camera Feeds
       |
       v
+----------------+     +---------------------+
| CameraManager  |---->|    FrameBroker       |  Pub/sub frame distribution
| (per-camera    |     | (memory-aware,       |  with priority-based dropping
|  threads)      |     |  drop monitoring)    |
+----------------+     +---------------------+
                              |
              +---------------+---------------+
              |               |               |
              v               v               v
     +--------------+  +------------+  +--------------+
     |   Motion     |  |  Display   |  |  Recording   |
     |  Detector    |  |  Buffer    |  |   Buffer     |
     | (Lucas-Kanade|  | (LRU, for  |  | (FIFO, for   |
     |  optical     |  |  web UI)   |  |  video clips)|
     |  flow)       |  +------------+  +--------------+
     +--------------+        |               |
           |                 v               v
           | motion     JPEG Encoding   VideoClipRecorder
           | detected   Worker          (pre/post event)
           v
     +--------------+
     |   YOLOv5     |  Separate process, shared memory,
     | (subprocess) |  GPU-accelerated when available
     +--------------+
           |
           v
     +--------------+     +-----------------+
     | AlertManager |---->| Telegram / Email|
     | (batching,   |     | Notifications   |
     |  cooldowns,  |     +-----------------+
     |  curfew)     |---->| SQLite Event Log|
     +--------------+     +-----------------+
           |
           v
     +--------------+
     | Flask-SocketIO|  Real-time dashboard
     | Web Dashboard |  with live feeds
     +--------------+
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Computer Vision | OpenCV, YOLOv5 (Ultralytics) |
| Deep Learning | PyTorch (CUDA 12.1 supported) |
| Web Server | Flask, Flask-SocketIO, Eventlet |
| Notifications | python-telegram-bot, yagmail |
| Database | SQLite |
| Concurrency | Threading, multiprocessing with shared memory |

## Project Structure

```
security_dashboard/
├── app.py               # Application entry point
├── core.py              # SecuritySystem orchestrator, AlertManager, VideoClipRecorder
├── camera_manager.py    # Multi-camera RTSP capture with auto-reconnection
├── motion_detector.py   # Lucas-Kanade optical flow motion detection
├── yolo_process.py      # Isolated YOLOv5 subprocess with shared memory IPC
├── frame_broker.py      # Pub/sub frame distribution with memory pressure awareness
├── frame_pool.py        # Reference-counted shared frame storage
├── memory_manager.py    # Bounded buffers with FIFO/LRU/Priority eviction
├── metrics.py           # Real-time metrics collection and threshold alerting
├── utils.py             # Configuration loading with .env secret management
└── web/
    ├── routes.py        # HTTP API and video streaming endpoints
    └── handlers.py      # SocketIO event handlers for live dashboard
```

## Getting Started

### Prerequisites

- Python 3.10+
- RTSP-capable IP cameras
- (Optional) NVIDIA GPU with CUDA 12.1 for accelerated inference

### Installation

```bash
git clone https://github.com/olibucko/argus.git
cd argus
pip install -r requirements.txt
```

### Configuration

1. Copy the example files:
   ```bash
   cp example_config.json config.json
   cp .env.example .env
   ```

2. Edit `.env` with your credentials (camera URLs, Telegram token, email):
   ```
   CAMERA_1_URL=rtsp://user:pass@192.168.1.100:554/stream1
   TELEGRAM_BOT_TOKEN=your_token_here
   ```

3. Edit `config.json` to tune detection parameters per camera — sensitivity, confidence thresholds, motion aggressiveness, YOLO interval, and curfew hours.

### Running

```bash
python -m security_dashboard.app
```

The dashboard will be available at `http://localhost:5000`.

## Design Decisions

**Motion-gated inference** — Running YOLO on every frame from every camera is computationally expensive. Argus uses lightweight Lucas-Kanade optical flow as a pre-filter, only submitting frames to YOLO when motion is detected. This reduces GPU/CPU load significantly while maintaining detection responsiveness.

**Subprocess isolation for YOLO** — Deep learning inference is isolated in a separate process using shared memory for frame transfer. This prevents inference latency from blocking the camera capture pipeline and allows the main process to remain responsive.

**Three-tier buffer architecture** — Each camera feed maintains separate display, processing, and recording buffers with different eviction policies (LRU, Priority, FIFO). This ensures the web UI stays smooth, detection processing prioritises active scenes, and video recording preserves temporal continuity.

**Batched alerting with cooldowns** — Detections are grouped into time-windowed batches to prevent notification spam. Curfew scheduling ensures alerts are only dispatched during configured hours.

## License

[MIT](LICENSE)
