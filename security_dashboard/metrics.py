"""
Centralized Metrics System - Real-time monitoring for frame processing pipeline.

This module provides comprehensive metrics collection, analysis, and alerting
for the security camera system's frame processing performance.
"""

import time
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Tuple
from enum import Enum
from flask import json
import logging

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class MetricValue:
    """A single metric measurement."""
    value: float
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricAlert:
    """Alert generated when metric exceeds threshold."""
    metric_name: str
    level: AlertLevel
    message: str
    value: float
    threshold: float
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class MetricThreshold:
    """Threshold configuration for metric alerting."""

    def __init__(self,
                 warning_threshold: Optional[float] = None,
                 error_threshold: Optional[float] = None,
                 critical_threshold: Optional[float] = None,
                 comparison: str = "greater"):  # "greater" or "less"
        self.warning_threshold = warning_threshold
        self.error_threshold = error_threshold
        self.critical_threshold = critical_threshold
        self.comparison = comparison

    def check_thresholds(self, value: float) -> Optional[AlertLevel]:
        """Check if value exceeds any threshold."""
        if self.comparison == "greater":
            if self.critical_threshold and value >= self.critical_threshold:
                return AlertLevel.CRITICAL
            elif self.error_threshold and value >= self.error_threshold:
                return AlertLevel.ERROR
            elif self.warning_threshold and value >= self.warning_threshold:
                return AlertLevel.WARNING
        else:  # "less"
            if self.critical_threshold and value <= self.critical_threshold:
                return AlertLevel.CRITICAL
            elif self.error_threshold and value <= self.error_threshold:
                return AlertLevel.ERROR
            elif self.warning_threshold and value <= self.warning_threshold:
                return AlertLevel.WARNING

        return None


class MetricCollector:
    """Collects and stores metrics with automatic aggregation."""

    def __init__(self, name: str, max_history: int = 1000):
        self.name = name
        self.max_history = max_history
        self.values: deque[MetricValue] = deque(maxlen=max_history)
        self.lock = threading.RLock()

        # Aggregation cache
        self._cached_avg = None
        self._cached_min = None
        self._cached_max = None
        self._cache_timestamp = 0
        self._cache_duration = 1.0  # Cache for 1 second

    def add_value(self, value: float, metadata: Dict[str, Any] = None) -> None:
        """Add a new metric value."""
        with self.lock:
            metric_value = MetricValue(
                value=value,
                metadata=metadata or {}
            )
            self.values.append(metric_value)
            self._invalidate_cache()

    def get_average(self, duration_seconds: Optional[float] = None) -> Optional[float]:
        """Get average value over specified duration."""
        with self.lock:
            if not self.values:
                return None

            if duration_seconds is None:
                # Use cached value if available
                if self._is_cache_valid():
                    return self._cached_avg

                values = [v.value for v in self.values]
                self._cached_avg = sum(values) / len(values)
                self._update_cache_timestamp()
                return self._cached_avg

            # Calculate for specific duration
            cutoff_time = time.time() - duration_seconds
            values = [v.value for v in self.values if v.timestamp >= cutoff_time]
            return sum(values) / len(values) if values else None

    def _is_cache_valid(self) -> bool:
        """Check if aggregation cache is still valid."""
        return (time.time() - self._cache_timestamp) < self._cache_duration

    def _invalidate_cache(self) -> None:
        """Invalidate aggregation cache."""
        self._cached_avg = None
        self._cached_min = None
        self._cached_max = None

    def _update_cache_timestamp(self) -> None:
        """Update cache timestamp."""
        self._cache_timestamp = time.time()


class SystemMetrics:
    """
    Centralized metrics system for the security camera application.

    Features:
    - Real-time metric collection and aggregation
    - Threshold-based alerting
    - Performance trend analysis
    - Web dashboard integration
    """

    def __init__(self):
        self.collectors: Dict[str, MetricCollector] = {}
        self.thresholds: Dict[str, MetricThreshold] = {}
        self.alerts: deque[MetricAlert] = deque(maxlen=100)
        self.alert_callbacks: List[Callable[[MetricAlert], None]] = []
        self.lock = threading.RLock()

        # Built-in metrics
        self._setup_builtin_metrics()

        logger.info("SystemMetrics initialized")

    def _setup_builtin_metrics(self) -> None:
        """Setup built-in metric collectors and thresholds."""

        # Frame processing metrics
        self.register_metric("frames_per_second")
        self.register_metric("frame_drop_rate")
        self.register_metric("processing_latency_ms")
        self.register_metric("queue_depth")
        self.register_metric("memory_usage_mb")

        # Camera metrics
        self.register_metric("camera_fps")
        self.register_metric("camera_frame_drops")
        self.register_metric("camera_reconnections")

        # YOLO metrics
        self.register_metric("yolo_inference_time_ms")
        self.register_metric("yolo_queue_depth")
        self.register_metric("yolo_accuracy_score")

        # Alert metrics
        self.register_metric("alerts_per_hour")
        self.register_metric("false_positive_rate")

        # System metrics
        self.register_metric("cpu_usage_percent")
        self.register_metric("system_uptime_hours")

        # Setup default thresholds
        self._setup_default_thresholds()

    def _setup_default_thresholds(self) -> None:
        """Setup default alert thresholds."""

        # Frame drop rate (percentage)
        self.set_threshold("frame_drop_rate", MetricThreshold(
            warning_threshold=5.0,    # 5% drop rate
            error_threshold=10.0,     # 10% drop rate
            critical_threshold=20.0,  # 20% drop rate
            comparison="greater"
        ))

        # Processing latency (milliseconds)
        self.set_threshold("processing_latency_ms", MetricThreshold(
            warning_threshold=100,    # 100ms latency
            error_threshold=250,      # 250ms latency
            critical_threshold=500,   # 500ms latency
            comparison="greater"
        ))

        # Memory usage (MB)
        self.set_threshold("memory_usage_mb", MetricThreshold(
            warning_threshold=800,    # 800MB usage
            error_threshold=1000,     # 1GB usage
            critical_threshold=1200,  # 1.2GB usage
            comparison="greater"
        ))

        # Queue depth
        self.set_threshold("queue_depth", MetricThreshold(
            warning_threshold=50,     # 50 items in queue
            error_threshold=80,       # 80 items in queue
            critical_threshold=100,   # 100 items in queue
            comparison="greater"
        ))

        # Camera FPS (frames per second)
        self.set_threshold("camera_fps", MetricThreshold(
            warning_threshold=15,     # Below 15 FPS
            error_threshold=10,       # Below 10 FPS
            critical_threshold=5,     # Below 5 FPS
            comparison="less"
        ))

    def register_metric(self, name: str, max_history: int = 1000) -> MetricCollector:
        """Register a new metric collector."""
        with self.lock:
            collector = MetricCollector(name, max_history)
            self.collectors[name] = collector
            logger.debug(f"Registered metric collector: {name}")
            return collector

    def record_metric(self, name: str, value: float, metadata: Dict[str, Any] = None) -> None:
        """Record a metric value."""
        with self.lock:
            if name not in self.collectors:
                self.register_metric(name)

            self.collectors[name].add_value(value, metadata)

            # Check thresholds
            if name in self.thresholds:
                self._check_threshold(name, value)

    def set_threshold(self, metric_name: str, threshold: MetricThreshold) -> None:
        """Set alert threshold for a metric."""
        with self.lock:
            self.thresholds[metric_name] = threshold
            logger.debug(f"Set threshold for metric: {metric_name}")

    def _check_threshold(self, metric_name: str, value: float) -> None:
        """Check if metric value exceeds threshold and generate alert if needed."""
        threshold = self.thresholds[metric_name]
        alert_level = threshold.check_thresholds(value)

        if alert_level:
            alert = MetricAlert(
                metric_name=metric_name,
                level=alert_level,
                message=f"{metric_name} is {alert_level.value}: {value}",
                value=value,
                threshold=self._get_threshold_value(threshold, alert_level)
            )

            self.alerts.append(alert)

            # Notify callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"Error in alert callback: {e}")

    def _get_threshold_value(self, threshold: MetricThreshold, level: AlertLevel) -> float:
        """Get the threshold value for a specific alert level."""
        if level == AlertLevel.WARNING:
            return threshold.warning_threshold
        elif level == AlertLevel.ERROR:
            return threshold.error_threshold
        elif level == AlertLevel.CRITICAL:
            return threshold.critical_threshold
        return 0.0

    def get_metric_summary(self, name: str, duration_seconds: float = 300) -> Dict[str, Any]:
        """Get comprehensive summary of a metric."""
        with self.lock:
            if name not in self.collectors:
                return {"error": f"Metric {name} not found"}

            collector = self.collectors[name]
            latest = collector.values[-1] if collector.values else None
            avg = collector.get_average(duration_seconds)
            
            # These methods were unused, so their logic is simplified or removed.
            min_val, max_val, rate = None, None, None
            return {
                "name": name,
                "latest_value": latest.value if latest else None,
                "latest_timestamp": latest.timestamp if latest else None,
                "average": avg,
                "min": min_val,
                "max": max_val,
                "rate_per_second": rate,
                "duration_seconds": duration_seconds,
                "sample_count": len(collector.values)
            }

    def get_all_metrics_summary(self, duration_seconds: float = 300) -> Dict[str, Dict[str, Any]]:
        """Get summary of all registered metrics."""
        with self.lock:
            summary = {}
            for name in self.collectors:
                summary[name] = self.get_metric_summary(name, duration_seconds)
            return summary

    def get_recent_alerts(self, max_alerts: int = 20) -> List[Dict[str, Any]]:
        """Get recent alerts as JSON-serializable data."""
        with self.lock:
            alerts = [self.alerts[i] for i in range(len(self.alerts) - 1, max(len(self.alerts) - max_alerts - 1, -1), -1)]
            return [
                {
                    "metric_name": alert.metric_name,
                    "level": alert.level.value,
                    "message": alert.message,
                    "value": alert.value,
                    "threshold": alert.threshold,
                    "timestamp": alert.timestamp,
                    "metadata": alert.metadata
                }
                for alert in alerts
            ]

    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        with self.lock:
            # Count alerts by level in last hour
            cutoff_time = time.time() - 3600  # 1 hour
            recent_alerts = [a for a in self.alerts if a.timestamp >= cutoff_time]

            alert_counts = defaultdict(int)
            for alert in recent_alerts:
                alert_counts[alert.level.value] += 1

            # Determine overall health status
            if alert_counts[AlertLevel.CRITICAL.value] > 0:
                health_status = "critical"
            elif alert_counts[AlertLevel.ERROR.value] > 0:
                health_status = "error"
            elif alert_counts[AlertLevel.WARNING.value] > 3:  # More than 3 warnings
                health_status = "warning"
            else:
                health_status = "healthy"

            return {
                "status": health_status,
                "alerts_last_hour": dict(alert_counts),
                "total_metrics": len(self.collectors),
                "active_thresholds": len(self.thresholds),
                "uptime_seconds": time.time() - self._start_time if hasattr(self, '_start_time') else 0
            }

    def shutdown(self) -> None:
        """Shutdown the metrics system."""
        logger.info("Shutting down SystemMetrics")
        with self.lock:
            self.collectors.clear()
            self.thresholds.clear()
            self.alerts.clear()
            self.alert_callbacks.clear()


# Global metrics instance
_global_metrics: Optional[SystemMetrics] = None
_metrics_lock = threading.Lock()


def get_global_metrics() -> SystemMetrics:
    """Get or create the global metrics instance."""
    global _global_metrics

    with _metrics_lock:
        if _global_metrics is None:
            _global_metrics = SystemMetrics()
            try:
                _global_metrics._start_time = time.time()
            except AttributeError:
                # This can happen in a race condition, but it's not critical.
                pass
        return _global_metrics


def record_metric(name: str, value: float, metadata: Dict[str, Any] = None) -> None:
    """Convenience function to record a metric value."""
    get_global_metrics().record_metric(name, value, metadata)