from flask import current_app
from flask_socketio import emit
from datetime import datetime
from typing import Dict
import time

from . import socketio
from ..utils import load_config, save_config

@socketio.on('connect')
def handle_connect():
    security_system = current_app.security_system
    if security_system:
        am = security_system.alert_manager
        emit('detection_state_update', {'enabled': am.detection_enabled})
        emit('curfew_times', {
            'start': am.config.curfew_start.strftime('%H:%M'),
            'end': am.config.curfew_end.strftime('%H:%M'),
            'cooldown_period': am.config.cooldown_period
        })
        # Send dynamic viewport configurations
        emit('viewport_configs', {
            'configs': security_system.viewport_configs,
            'grid_size': security_system.grid_size
        })

@socketio.on('set_detection_state')
def handle_set_detection_state(data: Dict[str, bool]):
    security_system = current_app.security_system
    if security_system:
        # Assumes an 'detection_enabled' attribute on AlertManager
        security_system.alert_manager.detection_enabled = data['enabled']
        socketio.emit('detection_state_update', {'enabled': security_system.alert_manager.detection_enabled})

@socketio.on('update_curfew_times')
def handle_curfew_times(data: Dict[str, str]):
    security_system = current_app.security_system
    if security_system:
        try:
            # Update the config file
            config = load_config()
            config['alert_manager']['curfew_start'] = data['start']
            config['alert_manager']['curfew_end'] = data['end']
            save_config(config)

            # Update the running instance
            am = security_system.alert_manager
            am.config.curfew_start = datetime.strptime(data['start'], '%H:%M').time()
            am.config.curfew_end = datetime.strptime(data['end'], '%H:%M').time()
            socketio.emit('curfew_times_updated', data)
        except ValueError as e:
            emit('error_message', {'message': f'Invalid time format: {e}'})

@socketio.on('update_viewport_config')
def handle_update_viewport_config(data: dict):
    """Handles updates to a specific viewport's configuration."""
    security_system = current_app.security_system
    try:
        row, col = data['row'], data['col']
        viewport_id_str = f"{row},{col}"
        
        viewport = security_system.get_viewport((row, col))
        if viewport:
            # Update the running instance
            viewport.update_config(data['config'])
            
            # Update the config by finding the camera and updating its settings
            config = load_config()

            # Find the camera associated with this viewport
            viewport_config = security_system.viewport_configs.get(viewport_id_str)
            if viewport_config:
                camera_name = viewport_config['camera_name']
                for camera in config['cameras']:
                    if camera['camera_name'] == camera_name:
                        camera.update(data['config'])
                        break
                save_config(config)

                # Update in-memory viewport_configs so it persists on refresh
                security_system.viewport_configs[viewport_id_str].update(data['config'])

            # Notify all clients of the change
            socketio.emit('viewport_config_updated', data)

    except (KeyError, TypeError) as e:
        emit('error_message', {'message': f'Invalid data format: {e}'})

@socketio.on('request_metrics')
def handle_request_metrics():
    """Handle request for real-time metrics data."""
    security_system = current_app.security_system
    if not security_system:
        emit('error_message', {'message': 'Security system not available'})
        return

    try:
        # Get comprehensive metrics
        metrics_summary = security_system.metrics.get_all_metrics_summary(duration_seconds=60)
        frame_broker_status = security_system.frame_broker.get_camera_status()
        memory_stats = security_system.memory_manager.get_global_stats()
        health_status = security_system.metrics.get_system_health()

        # Send metrics to requesting client
        emit('metrics_update', {
            'metrics': metrics_summary,
            'frame_broker': frame_broker_status,
            'memory': memory_stats,
            'health': health_status,
            'timestamp': time.time()
        })
    except Exception as e:
        emit('error_message', {'message': f'Failed to get metrics: {str(e)}'})

@socketio.on('request_buffer_stats')
def handle_request_buffer_stats():
    """Handle request for memory buffer statistics."""
    security_system = current_app.security_system
    if not security_system:
        emit('error_message', {'message': 'Security system not available'})
        return

    try:
        buffer_stats = {}

        # Collect all buffer stats
        for vp_id, buffer in security_system.viewport_buffers.items():
            buffer_stats[f"viewport_{vp_id[0]}_{vp_id[1]}"] = buffer.get_stats()

        for vp_id, buffer in security_system.recording_buffers.items():
            buffer_stats[f"recording_{vp_id[0]}_{vp_id[1]}"] = buffer.get_stats()

        for vp_id, buffer in security_system.display_buffers.items():
            buffer_stats[f"display_{vp_id[0]}_{vp_id[1]}"] = buffer.get_stats()

        emit('buffer_stats_update', {'buffer_stats': buffer_stats})
    except Exception as e:
        emit('error_message', {'message': f'Failed to get buffer stats: {str(e)}'})

@socketio.on('clear_metrics_alerts')
def handle_clear_metrics_alerts():
    """Handle request to clear system alerts."""
    security_system = current_app.security_system
    if security_system:
        security_system.metrics.clear_alerts()
        socketio.emit('alerts_cleared')