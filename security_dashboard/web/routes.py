from flask import Blueprint, Response, jsonify, send_from_directory, render_template, current_app, request
from dataclasses import asdict
import time
import sqlite3
import os
import eventlet

from ..utils import DB_FILE

api_bp = Blueprint('api', __name__)

@api_bp.route('/')
def index():
    return render_template('index_deployment.html')

@api_bp.route('/event_videos/<path:filename>')
def serve_video(filename):
    """Serve video files with range request support for browser seeking."""
    video_directory = os.path.join(current_app.config['PROJECT_ROOT'], 'event_videos')
    video_path = os.path.abspath(os.path.join(video_directory, filename))

    # Security: Ensure the resolved path is within the video directory (prevent path traversal)
    if not video_path.startswith(os.path.abspath(video_directory)):
        return jsonify({'error': 'Invalid file path'}), 403

    # Validate file exists
    if not os.path.isfile(video_path):
        return jsonify({'error': 'Video not found'}), 404

    # Get file size
    file_size = os.path.getsize(video_path)

    # Determine content type based on file extension
    ext = os.path.splitext(filename)[1].lower()
    content_type_map = {
        '.mp4': 'video/mp4',
        '.avi': 'video/x-msvideo',
        '.mov': 'video/quicktime',
        '.mkv': 'video/x-matroska',
        '.webm': 'video/webm',
        '.m4v': 'video/mp4'
    }
    content_type = content_type_map.get(ext, 'video/mp4')

    # Parse range header
    range_header = request.headers.get('Range')

    if not range_header:
        # No range requested, serve entire file using a generator for memory efficiency
        def generate():
            with open(video_path, 'rb') as f:
                chunk_size = 8192  # 8KB chunks
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk

        response = Response(generate(), 200, mimetype=content_type)
        response.headers.add('Accept-Ranges', 'bytes')
        response.headers.add('Content-Length', str(file_size))
        return response

    # Parse range header (format: "bytes=start-end")
    try:
        range_match = range_header.replace('bytes=', '').split('-')
        start = int(range_match[0]) if range_match[0] else 0
        end = int(range_match[1]) if range_match[1] else file_size - 1

        # Validate range
        if start >= file_size or start < 0:
            return Response(status=416)  # Range Not Satisfiable

        # Ensure end doesn't exceed file size
        end = min(end, file_size - 1)
        length = end - start + 1

        # Read the requested range
        with open(video_path, 'rb') as f:
            f.seek(start)
            data = f.read(length)

        # Create response with 206 Partial Content
        response = Response(data, 206, mimetype=content_type)
        response.headers.add('Content-Range', f'bytes {start}-{end}/{file_size}')
        response.headers.add('Accept-Ranges', 'bytes')
        response.headers.add('Content-Length', str(length))

        return response

    except (ValueError, IndexError):
        # Invalid range header format
        return Response(status=400)

@api_bp.route('/api/events')
def get_events():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT timestamp, viewport_name, confidence, video_path FROM events ORDER BY timestamp DESC")
    events = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return jsonify(events)

@api_bp.route('/api/cameras/status')
def get_camera_status():
    """Get status of all cameras."""
    security_system = current_app.security_system
    if not security_system:
        return jsonify({'error': 'Security system not available'}), 500
    
    camera_status = security_system.get_camera_status()
    
    # Format the response for the frontend
    formatted_status = {}
    for camera_name, status in camera_status.items():
        formatted_status[camera_name] = {
            'connected': status['connected'],
            'frame_count': status['frame_count'],
            'last_frame_time': status['last_frame_time'],
            'last_frame_ago': time.time() - status['last_frame_time'] if status['last_frame_time'] > 0 else 0,
            'camera_id': status['config'].camera_id,
            'resolution': f"{status['config'].width}x{status['config'].height}",
            'fps': status['config'].fps,
            'enabled': status['config'].enabled
        }
    
    return jsonify(formatted_status)

@api_bp.route('/api/cameras/<camera_name>/restart', methods=['POST'])
def restart_camera(camera_name: str):
    """Restart a specific camera."""
    security_system = current_app.security_system
    if not security_system:
        return jsonify({'error': 'Security system not available'}), 500
    
    success = security_system.restart_camera(camera_name)
    return jsonify({'success': success, 'camera_name': camera_name})

@api_bp.route('/video_feed/<int:row>/<int:col>')
def video_feed(row: int, col: int):
    security_system = current_app.security_system

    def generate_frames(viewport_id: tuple):
        """Stream pre-encoded JPEG frames to clients."""
        # Wait for first frame to be available (up to 5 seconds)
        max_wait = 5.0
        wait_start = time.time()
        first_frame_sent = False

        # DEBUG: Log what's in encoded_frames when this route starts
        print(f"[VIDEO_FEED] Request for viewport {viewport_id}")
        print(f"[VIDEO_FEED] Available encoded frames: {list(security_system.encoded_frames.keys())}")

        while True:
            # Get pre-encoded JPEG frame (no encoding needed per client!)
            encoded_frame = security_system.get_encoded_frame(viewport_id)
            if encoded_frame is not None:
                if not first_frame_sent:
                    print(f"[VIDEO_FEED] First frame sent for viewport {viewport_id}")
                try:
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + encoded_frame + b'\r\n')
                except (BrokenPipeError, ConnectionResetError):
                    # Client disconnected, exit the generator cleanly
                    print(f"[VIDEO_FEED] Client disconnected for viewport {viewport_id}")
                    return

                first_frame_sent = True
            elif not first_frame_sent and (time.time() - wait_start) < max_wait:
                # Still waiting for first frame, don't give up yet
                eventlet.sleep(0.1)  # Use eventlet.sleep to yield to other green threads
                continue
            elif not first_frame_sent:
                # Timeout - never got a frame
                print(f"[VIDEO_FEED] TIMEOUT for viewport {viewport_id} - no frames after {max_wait}s")
                print(f"[VIDEO_FEED] Final check - available frames: {list(security_system.encoded_frames.keys())}")
                return  # Exit generator to close connection

            eventlet.sleep(1/20)  # Use eventlet.sleep to yield to other green threads

    return Response(generate_frames((row, col)), mimetype='multipart/x-mixed-replace; boundary=frame')

@api_bp.route('/viewport_config/<int:row>/<int:col>')
def get_viewport_config(row: int, col: int):
    security_system = current_app.security_system
    viewport = security_system.get_viewport((row, col))
    if viewport:
        return jsonify(asdict(viewport.config))
    return jsonify({'error': 'Config not found'}), 404

@api_bp.route('/api/viewport_configs')
def get_all_viewport_configs():
    """Get all viewport configurations and the grid size."""
    security_system = current_app.security_system
    if not security_system:
        return jsonify({'error': 'Security system not available'}), 500
    return jsonify({
        'configs': security_system.viewport_configs,
        'grid_size': security_system.grid_size
    })

@api_bp.route('/api/metrics/summary')
def get_metrics_summary():
    """Get comprehensive system metrics summary."""
    security_system = current_app.security_system
    if not security_system:
        return jsonify({'error': 'Security system not available'}), 500

    try:
        metrics_summary = security_system.metrics.get_all_metrics_summary(duration_seconds=300)
        frame_broker_status = security_system.frame_broker.get_camera_status()
        memory_stats = security_system.memory_manager.get_global_stats()
        health_status = security_system.metrics.get_system_health()

        return jsonify({
            'metrics': metrics_summary,
            'frame_broker': frame_broker_status,
            'memory': memory_stats,
            'health': health_status,
            'timestamp': time.time()
        })
    except Exception as e:
        return jsonify({'error': f'Failed to get metrics: {str(e)}'}), 500

@api_bp.route('/api/metrics/alerts')
def get_recent_alerts():
    """Get recent system alerts."""
    security_system = current_app.security_system
    if not security_system:
        return jsonify({'error': 'Security system not available'}), 500

    try:
        alerts = security_system.metrics.get_recent_alerts(max_alerts=50)
        return jsonify({'alerts': alerts})
    except Exception as e:
        return jsonify({'error': f'Failed to get alerts: {str(e)}'}), 500

@api_bp.route('/api/metrics/<metric_name>')
def get_metric_detail(metric_name: str):
    """Get detailed information about a specific metric."""
    security_system = current_app.security_system
    if not security_system:
        return jsonify({'error': 'Security system not available'}), 500

    duration = request.args.get('duration', 300, type=int)

    try:
        metric_summary = security_system.metrics.get_metric_summary(metric_name, duration)
        return jsonify(metric_summary)
    except Exception as e:
        return jsonify({'error': f'Failed to get metric {metric_name}: {str(e)}'}), 500

@api_bp.route('/api/buffer_stats')
def get_buffer_stats():
    """Get memory buffer statistics."""
    security_system = current_app.security_system
    if not security_system:
        return jsonify({'error': 'Security system not available'}), 500

    try:
        buffer_stats = {}

        # Collect viewport buffer stats
        for vp_id, buffer in security_system.viewport_buffers.items():
            buffer_stats[f"viewport_{vp_id[0]}_{vp_id[1]}"] = buffer.get_stats()

        # Collect recording buffer stats
        for vp_id, buffer in security_system.recording_buffers.items():
            buffer_stats[f"recording_{vp_id[0]}_{vp_id[1]}"] = buffer.get_stats()

        # Collect display buffer stats
        for vp_id, buffer in security_system.display_buffers.items():
            buffer_stats[f"display_{vp_id[0]}_{vp_id[1]}"] = buffer.get_stats()

        return jsonify({'buffer_stats': buffer_stats})
    except Exception as e:
        return jsonify({'error': f'Failed to get buffer stats: {str(e)}'}), 500

@api_bp.route('/api/email/test', methods=['POST'])
def test_email():
    """Send a test email to verify configuration."""
    security_system = current_app.security_system
    if not security_system:
        return jsonify({'error': 'Security system not available'}), 500

    alert_manager = security_system.alert_manager

    if not alert_manager.email_enabled:
        return jsonify({
            'success': False,
            'error': 'Email is not enabled. Check your email credentials in config.json'
        }), 400

    try:
        from datetime import datetime
        test_content = f"""
        <!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"></head>
        <body style="margin: 0; padding: 20px; background-color: #09090b; font-family: Arial, sans-serif;">
            <div style="max-width: 600px; margin: 0 auto; background-color: #18181b; border: 1px solid #27272a; border-radius: 12px; padding: 30px;">
                <h1 style="color: #fafafa; margin: 0 0 20px 0; font-size: 24px;">Argus SECURITY - TEST EMAIL</h1>
                <p style="color: #d4d4d8; font-size: 16px; line-height: 1.6;">
                    This is a test email from your security system.
                    <br><br>
                    <strong>Test Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                    <br><br>
                    If you received this email, your email configuration is working correctly!
                </p>
                <p style="color: #71717a; font-size: 12px; margin-top: 30px; padding-top: 20px; border-top: 1px solid #27272a;">
                    This is an automated test from the Argus Security System.
                </p>
            </div>
        </body></html>
        """

        alert_manager._send_email(
            subject="Security System - Test Email",
            content=test_content,
            attachments=None
        )

        return jsonify({
            'success': True,
            'message': f'Test email sent to {alert_manager.config.recipient_list}'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to send test email: {str(e)}'
        }), 500

@api_bp.route('/api/email/status')
def email_status():
    """Get email configuration status."""
    security_system = current_app.security_system
    if not security_system:
        return jsonify({'error': 'Security system not available'}), 500

    alert_manager = security_system.alert_manager

    return jsonify({
        'enabled': alert_manager.email_enabled,
        'sender': alert_manager.config.sender_email if alert_manager.config.sender_email else None,
        'recipients': alert_manager.config.recipient_list if alert_manager.config.recipient_list else [],
        'curfew_start': alert_manager.config.curfew_start.strftime('%H:%M'),
        'curfew_end': alert_manager.config.curfew_end.strftime('%H:%M'),
        'is_curfew_hours': alert_manager.is_curfew_hours()
    })

@api_bp.route('/api/telegram/test', methods=['POST'])
def test_telegram():
    """Send a test Telegram message to verify configuration."""
    security_system = current_app.security_system
    if not security_system:
        return jsonify({'error': 'Security system not available'}), 500

    alert_manager = security_system.alert_manager

    if not alert_manager.telegram_enabled:
        return jsonify({
            'success': False,
            'error': 'Telegram is not enabled. Check your bot token and chat IDs in config.json'
        }), 400

    try:
        from datetime import datetime
        test_message = f"🤖 *SECURITY SYSTEM TEST*\n\n"
        test_message += f"This is a test message from your security system.\n\n"
        test_message += f"*Test Time:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        test_message += f"If you received this message, your Telegram configuration is working correctly!\n\n"
        test_message += f"_Argus Security System_"

        # Send test message
        success = alert_manager._send_telegram_alert(test_message, screenshot_path=None)

        if success:
            return jsonify({
                'success': True,
                'message': f'Test message sent to {len(alert_manager.config.telegram_chat_ids)} chat(s)'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to send test message. Check logs for details.'
            }), 500

    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to send test message: {str(e)}'
        }), 500

@api_bp.route('/api/telegram/status')
def telegram_status():
    """Get Telegram bot configuration status."""
    security_system = current_app.security_system
    if not security_system:
        return jsonify({'error': 'Security system not available'}), 500

    alert_manager = security_system.alert_manager

    bot_info = None
    if alert_manager.telegram_bot and alert_manager.telegram_enabled:
        try:
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                bot_data = loop.run_until_complete(alert_manager.telegram_bot.get_me())
                bot_info = {
                    'username': bot_data.username,
                    'first_name': bot_data.first_name,
                    'id': bot_data.id
                }
            finally:
                loop.close()
        except Exception as e:
            bot_info = {'error': str(e)}

    return jsonify({
        'enabled': alert_manager.telegram_enabled,
        'bot_info': bot_info,
        'chat_ids': alert_manager.config.telegram_chat_ids if alert_manager.config.telegram_chat_ids else [],
        'chat_count': len(alert_manager.config.telegram_chat_ids) if alert_manager.config.telegram_chat_ids else 0,
        'email_as_fallback': alert_manager.config.email_as_fallback,
        'curfew_start': alert_manager.config.curfew_start.strftime('%H:%M'),
        'curfew_end': alert_manager.config.curfew_end.strftime('%H:%M'),
        'is_curfew_hours': alert_manager.is_curfew_hours()
    })