# rpi_frontend.py - Raspberry Pi Frontend (Web UI + Camera)
from flask import Flask, Response, jsonify, render_template_string
import cv2
import numpy as np
import time
import socket
import requests
import base64
import io
from PIL import Image
import subprocess

app = Flask(__name__)

# ==================== CONFIGURATION ====================
WINDOWS_IP = "192.168.43.82"  # ← CHANGE THIS TO YOUR WINDOWS IP!
WINDOWS_ML_SERVER = f"http://{WINDOWS_IP}:5000"
WINDOWS_SENSOR_SERVER = f"http://{WINDOWS_IP}:5001"

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
FRAME_SKIP = 2

# ==================== PI CAMERA FUNCTIONS ====================
def capture_pi_camera():
    try:
        result = subprocess.run([
            'rpicam-still', '-o', '-', 
            '--width', str(CAMERA_WIDTH), 
            '--height', str(CAMERA_HEIGHT),
            '--nopreview', 
            '--encoding', 'jpg', 
            '--timeout', '1'
        ], capture_output=True, timeout=3)
        
        if result.stdout and len(result.stdout) > 1000:
            img_bytes = io.BytesIO(result.stdout)
            img = Image.open(img_bytes)
            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            return frame
    except Exception as e:
        print(f"Pi Camera error: {e}")
    return None

def capture_pi_camera_continuous():
    while True:
        frame = capture_pi_camera()
        if frame is not None:
            yield frame
        else:
            blank = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
            cv2.putText(blank, "Camera Error - Check Connection", (50, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            yield blank
        time.sleep(0.05)

# ==================== FETCH SENSORS FROM WINDOWS ====================
def fetch_sensors_from_windows():
    """Fetch sensor data from Windows Node.js server"""
    try:
        response = requests.get(f"{WINDOWS_SENSOR_SERVER}/api/sensors/all", timeout=2)
        if response.status_code == 200:
            data = response.json()
            if data.get('success', False):
                return data.get('data', {})
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Windows sensor server!")
        print("   Make sure sensor_server.js is running on Windows")
    except Exception as e:
        print(f"❌ Error fetching sensors: {e}")
    return {}

# ==================== ML PROCESSING ====================
def send_to_ml_server(frame):
    try:
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        response = requests.post(
            f"{WINDOWS_ML_SERVER}/api/detect",
            json={'image': image_base64},
            timeout=5
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success', False):
                annotated_base64 = data.get('annotated_image')
                if annotated_base64:
                    annotated_bytes = base64.b64decode(annotated_base64)
                    annotated_image = Image.open(io.BytesIO(annotated_bytes))
                    annotated_frame = cv2.cvtColor(np.array(annotated_image), cv2.COLOR_RGB2BGR)
                    return annotated_frame, data.get('counts', {"total_fish": 0})
    except Exception as e:
        print(f"Error sending to ML server: {e}")
    
    error_frame = frame.copy()
    cv2.putText(error_frame, "ML Server Error", (50, 240), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return error_frame, {"guppy": 0, "molly": 0, "platy": 0, "total_fish": 0}

# ==================== GLOBAL VARIABLES ====================
frame_counter = 0
last_annotated_frame = None
current_counts = {"guppy": 0, "molly": 0, "platy": 0, "total_fish": 0}
current_sensors = {}
sensor_server_status = "Checking..."
ml_server_status = "Checking..."

# ==================== MJPEG STREAM GENERATOR ====================
def generate_frames():
    global frame_counter, last_annotated_frame, current_counts, current_sensors, sensor_server_status, ml_server_status
    
    camera_generator = capture_pi_camera_continuous()
    
    for frame in camera_generator:
        frame_counter += 1
        
        if frame_counter % FRAME_SKIP == 0:
            # ML detection
            annotated_frame, counts = send_to_ml_server(frame)
            if annotated_frame is not None:
                last_annotated_frame = annotated_frame
                current_counts = counts
                ml_server_status = "Connected"
            else:
                ml_server_status = "Disconnected"
            
            # Fetch sensor data every 10 frames
            if frame_counter % 10 == 0:
                sensor_data = fetch_sensors_from_windows()
                if sensor_data:
                    current_sensors = sensor_data
                    sensor_server_status = "Connected"
                else:
                    sensor_server_status = "Disconnected"
        else:
            annotated_frame = last_annotated_frame if last_annotated_frame is not None else frame
        
        if annotated_frame is not None:
            # Add sensor data overlay
            if current_sensors:
                ph = current_sensors.get('ph', 0)
                tds = current_sensors.get('tds', 0)
                light = current_sensors.get('light', 0)
                mq135 = current_sensors.get('mq135', 0)
                ph_status = current_sensors.get('ph_status', 'NEUTRAL')
                float_state = current_sensors.get('float_state', 'UNKNOWN')
                data_count = current_sensors.get('data_count', 0)
                
                cv2.putText(annotated_frame, f"ML: {ml_server_status} | Sensors: {sensor_server_status}", (10, 110), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 200, 255), 1)
                cv2.putText(annotated_frame, f"pH: {ph:.2f} ({ph_status})  TDS: {tds}ppm", (10, 130), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 100), 1)
                cv2.putText(annotated_frame, f"Light: {light}lx  MQ135: {mq135}", (10, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 100), 1)
                cv2.putText(annotated_frame, f"Float: {float_state}  Data: {data_count}", (10, 170), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
            
            ret, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
        time.sleep(0.05)

# ==================== ROUTES ====================

@app.route('/stream.mjpg')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/counts')
def get_counts():
    return jsonify(current_counts)

@app.route('/api/sensors')
def get_sensors():
    """Get sensor data from Windows Node.js server"""
    data = fetch_sensors_from_windows()
    if data:
        current_sensors.update(data)
    return jsonify(current_sensors)

@app.route('/api/sensors/status')
def get_sensor_status():
    """Check if Windows sensor server is running"""
    try:
        response = requests.get(f"{WINDOWS_SENSOR_SERVER}/api/sensors/status", timeout=2)
        return jsonify(response.json())
    except:
        return jsonify({"serial_connected": False, "error": "Windows server not running"})

@app.route('/api/status')
def get_status():
    """Get system status"""
    return jsonify({
        'ml_server': ml_server_status,
        'sensor_server': sensor_server_status,
        'windows_ip': WINDOWS_IP,
        'pi_ip': socket.gethostbyname(socket.gethostname())
    })

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE, windows_ip=WINDOWS_IP)

# ==================== HTML TEMPLATE ====================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AquaPonics - Smart Fish Monitoring</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            background: linear-gradient(135deg, #0c1a1a 0%, #1a3a3a 50%, #0c1a1a 100%);
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            color: #e0e0e0;
            padding: 20px;
            min-height: 100vh;
        }
        
        .container {
            max-width: 1600px;
            margin: 0 auto;
        }
        
        /* Header */
        .header {
            background: linear-gradient(135deg, rgba(0, 40, 40, 0.9), rgba(0, 60, 60, 0.9));
            border-radius: 20px;
            padding: 25px 35px;
            margin-bottom: 25px;
            border: 1px solid rgba(46, 204, 113, 0.2);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
        }
        
        .header-left {
            display: flex;
            align-items: center;
            gap: 20px;
        }
        
        .header-left h1 {
            color: #2ecc71;
            font-size: 2rem;
            font-weight: 700;
            letter-spacing: 1px;
        }
        
        .header-left h1 i {
            margin-right: 12px;
            color: #2ecc71;
        }
        
        .header-badge {
            background: rgba(46, 204, 113, 0.15);
            padding: 6px 16px;
            border-radius: 20px;
            font-size: 0.75rem;
            border: 1px solid rgba(46, 204, 113, 0.3);
        }
        
        .header-badge i {
            color: #2ecc71;
            margin-right: 6px;
        }
        
        .header-right {
            display: flex;
            align-items: center;
            gap: 15px;
            flex-wrap: wrap;
        }
        
        .status-indicator {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 16px;
            border-radius: 12px;
            font-size: 0.85rem;
            background: rgba(0, 0, 0, 0.3);
        }
        
        .status-indicator .dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            display: inline-block;
        }
        
        .dot-online {
            background: #2ecc71;
            animation: pulse 1.5s infinite;
        }
        
        .dot-offline {
            background: #e74c3c;
        }
        
        @keyframes pulse {
            0% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.6; transform: scale(1.2); }
            100% { opacity: 1; transform: scale(1); }
        }
        
        .status-text {
            color: #b0b0b0;
        }
        
        .status-text .online {
            color: #2ecc71;
        }
        
        .status-text .offline {
            color: #e74c3c;
        }
        
        /* Dashboard Grid */
        .dashboard {
            display: grid;
            grid-template-columns: 280px 1fr 300px;
            gap: 20px;
        }
        
        @media (max-width: 1200px) {
            .dashboard {
                grid-template-columns: 1fr;
            }
        }
        
        /* Panels */
        .panel {
            background: rgba(0, 30, 30, 0.85);
            backdrop-filter: blur(10px);
            border-radius: 16px;
            padding: 22px;
            border: 1px solid rgba(46, 204, 113, 0.15);
            box-shadow: 0 4px 24px rgba(0, 0, 0, 0.3);
            transition: border-color 0.3s ease;
        }
        
        .panel:hover {
            border-color: rgba(46, 204, 113, 0.3);
        }
        
        .panel-title {
            font-size: 1.1rem;
            font-weight: 600;
            color: #2ecc71;
            border-bottom: 2px solid rgba(46, 204, 113, 0.2);
            padding-bottom: 12px;
            margin-bottom: 18px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .panel-title i {
            font-size: 1.2rem;
        }
        
        /* Fish Counter */
        .fish-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 0;
            border-bottom: 1px solid rgba(46, 204, 113, 0.08);
        }
        
        .fish-item:last-child {
            border-bottom: none;
        }
        
        .fish-name {
            display: flex;
            align-items: center;
            gap: 10px;
            color: #c0d0d0;
        }
        
        .fish-name i {
            font-size: 1.2rem;
        }
        
        .fish-color-guppy { color: #2ecc71; }
        .fish-color-molly { color: #3498db; }
        .fish-color-platy { color: #f1c40f; }
        
        .fish-count {
            font-size: 1.4rem;
            font-weight: 700;
            color: #2ecc71;
            background: rgba(46, 204, 113, 0.1);
            padding: 2px 16px;
            border-radius: 20px;
            min-width: 40px;
            text-align: center;
        }
        
        .total-fish {
            margin-top: 16px;
            padding: 16px;
            background: linear-gradient(135deg, rgba(46, 204, 113, 0.2), rgba(46, 204, 113, 0.05));
            border-radius: 12px;
            border: 1px solid rgba(46, 204, 113, 0.2);
            text-align: center;
        }
        
        .total-fish .label {
            font-size: 0.9rem;
            color: #a0b0b0;
        }
        
        .total-fish .count {
            font-size: 2.8rem;
            font-weight: 800;
            color: #2ecc71;
            display: block;
            margin-top: 4px;
        }
        
        /* Camera */
        .camera-container {
            background: #000;
            border-radius: 12px;
            overflow: hidden;
            position: relative;
        }
        
        .camera-container img {
            width: 100%;
            display: block;
        }
        
        .camera-overlay {
            position: absolute;
            bottom: 12px;
            left: 12px;
            right: 12px;
            display: flex;
            justify-content: space-between;
            font-size: 0.7rem;
            color: rgba(255, 255, 255, 0.6);
            background: rgba(0, 0, 0, 0.5);
            padding: 6px 12px;
            border-radius: 8px;
            backdrop-filter: blur(4px);
        }
        
        /* Sensors */
        .sensor-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 0;
            border-bottom: 1px solid rgba(46, 204, 113, 0.06);
        }
        
        .sensor-item:last-child {
            border-bottom: none;
        }
        
        .sensor-label {
            display: flex;
            align-items: center;
            gap: 10px;
            color: #b0c0c0;
            font-size: 0.9rem;
        }
        
        .sensor-label i {
            width: 20px;
            color: #2ecc71;
        }
        
        .sensor-value {
            font-weight: 600;
            color: #e0e0e0;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        
        .sensor-value .unit {
            font-weight: 400;
            color: #808080;
            font-size: 0.75rem;
        }
        
        .sensor-badge {
            font-size: 0.65rem;
            padding: 2px 10px;
            border-radius: 12px;
            font-weight: 600;
        }
        
        .badge-neutral {
            background: rgba(46, 204, 113, 0.2);
            color: #2ecc71;
        }
        
        .badge-acidic {
            background: rgba(231, 76, 60, 0.2);
            color: #e74c3c;
        }
        
        .badge-alkaline {
            background: rgba(241, 196, 15, 0.2);
            color: #f1c40f;
        }
        
        .sensor-item.mq135-highlight {
            background: rgba(46, 204, 113, 0.05);
            border-radius: 8px;
            padding: 10px 12px;
            margin: 4px 0;
        }
        
        /* Float State */
        .float-state {
            font-weight: 600;
        }
        
        .float-ok {
            color: #2ecc71;
        }
        
        .float-low {
            color: #e74c3c;
        }
        
        .float-unknown {
            color: #f39c12;
        }
        
        /* Responsive */
        @media (max-width: 768px) {
            .header {
                flex-direction: column;
                align-items: flex-start;
                gap: 12px;
            }
            
            .header-right {
                width: 100%;
                justify-content: flex-start;
            }
            
            .panel {
                padding: 16px;
            }
        }
        
        /* Scrollbar */
        ::-webkit-scrollbar {
            width: 6px;
        }
        
        ::-webkit-scrollbar-track {
            background: rgba(0, 0, 0, 0.2);
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: #2ecc71;
            border-radius: 10px;
        }
        
        /* Glow effect for important values */
        .glow-text {
            text-shadow: 0 0 20px rgba(46, 204, 113, 0.3);
        }
        
        /* ML Status indicator in camera */
        .ml-indicator {
            position: absolute;
            top: 12px;
            left: 12px;
            background: rgba(0, 0, 0, 0.7);
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.7rem;
            color: #2ecc71;
            backdrop-filter: blur(4px);
            border: 1px solid rgba(46, 204, 113, 0.2);
        }
        
        .ml-indicator i {
            margin-right: 4px;
        }
        
        .ml-indicator.offline {
            color: #e74c3c;
            border-color: rgba(231, 76, 60, 0.3);
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <div class="header-left">
                <h1><i class="fas fa-fish"></i> AquaPonics</h1>
                <span class="header-badge">
                    <i class="fas fa-microchip"></i> v2.0
                </span>
            </div>
            <div class="header-right">
                <div class="status-indicator">
                    <span class="dot dot-online" id="statusDot"></span>
                    <span class="status-text">
                        <span id="statusLabel">Connected</span>
                    </span>
                </div>
                <div class="status-indicator" style="font-size:0.75rem; color:#888;">
                    <i class="fas fa-server"></i>
                    <span id="serverInfo">{{ windows_ip }}</span>
                </div>
            </div>
        </div>
        
        <!-- Dashboard -->
        <div class="dashboard">
            <!-- Left Panel - Fish Counter -->
            <div class="panel">
                <div class="panel-title">
                    <i class="fas fa-chart-simple"></i> Fish Counter
                </div>
                
                <div class="fish-item">
                    <span class="fish-name">
                        <i class="fas fa-fish fish-color-guppy"></i> Guppy
                    </span>
                    <span class="fish-count" id="guppy">0</span>
                </div>
                
                <div class="fish-item">
                    <span class="fish-name">
                        <i class="fas fa-fish fish-color-molly"></i> Molly
                    </span>
                    <span class="fish-count" id="molly">0</span>
                </div>
                
                <div class="fish-item">
                    <span class="fish-name">
                        <i class="fas fa-fish fish-color-platy"></i> Platy
                    </span>
                    <span class="fish-count" id="platy">0</span>
                </div>
                
                <div class="total-fish">
                    <span class="label"><i class="fas fa-flag-checkered"></i> TOTAL FISH</span>
                    <span class="count" id="total">0</span>
                </div>
            </div>
            
            <!-- Center Panel - Camera Feed -->
            <div class="panel">
                <div class="panel-title">
                    <i class="fas fa-video"></i> Live Feed
                </div>
                <div class="camera-container">
                    <img id="cameraFeed" src="/stream.mjpg" alt="Live Camera Feed">
                    <div class="ml-indicator" id="mlIndicator">
                        <i class="fas fa-brain"></i> <span id="mlStatusText">Active</span>
                    </div>
                    <div class="camera-overlay">
                        <span><i class="fas fa-circle" style="color:#2ecc71; font-size:8px;"></i> REC</span>
                        <span><i class="fas fa-clock"></i> <span id="timeDisplay">--:--:--</span></span>
                    </div>
                </div>
            </div>
            
            <!-- Right Panel - Sensors -->
            <div class="panel">
                <div class="panel-title">
                    <i class="fas fa-microchip"></i> ESP32 Sensors
                </div>
                
                <!-- pH -->
                <div class="sensor-item">
                    <span class="sensor-label">
                        <i class="fas fa-vial"></i> pH Level
                    </span>
                    <span class="sensor-value">
                        <span id="ph">0.00</span>
                        <span class="unit">pH</span>
                        <span class="sensor-badge badge-neutral" id="phStatus">NEUTRAL</span>
                    </span>
                </div>
                
                <!-- TDS -->
                <div class="sensor-item">
                    <span class="sensor-label">
                        <i class="fas fa-droplet"></i> TDS
                    </span>
                    <span class="sensor-value">
                        <span id="tds">0</span>
                        <span class="unit">ppm</span>
                    </span>
                </div>
                
                <!-- Light -->
                <div class="sensor-item">
                    <span class="sensor-label">
                        <i class="fas fa-sun"></i> Light
                    </span>
                    <span class="sensor-value">
                        <span id="light">0</span>
                        <span class="unit">lux</span>
                    </span>
                </div>
                
                <!-- MQ135 - Highlighted -->
                <div class="sensor-item mq135-highlight">
                    <span class="sensor-label">
                        <i class="fas fa-wind"></i> Air Quality
                    </span>
                    <span class="sensor-value">
                        <span id="mq135">0</span>
                        <span class="unit">ADC</span>
                        <span class="sensor-badge badge-neutral" id="mq135Status">GOOD</span>
                    </span>
                </div>
                
                <!-- Float Switch -->
                <div class="sensor-item">
                    <span class="sensor-label">
                        <i class="fas fa-water"></i> Float Switch
                    </span>
                    <span class="sensor-value float-unknown" id="floatState">UNKNOWN</span>
                </div>
                
                <!-- Data Count -->
                <div class="sensor-item">
                    <span class="sensor-label">
                        <i class="fas fa-database"></i> Data Packets
                    </span>
                    <span class="sensor-value">
                        <span id="dataCount">0</span>
                    </span>
                </div>
                
                <!-- Last Update -->
                <div class="sensor-item" style="border-bottom: none; padding-bottom: 0;">
                    <span class="sensor-label">
                        <i class="fas fa-clock"></i> Last Update
                    </span>
                    <span class="sensor-value" style="font-weight:400; color:#888; font-size:0.85rem;" id="lastUpdate">Never</span>
                </div>
            </div>
        </div>
    </div>

    <script>
        // ==================== TIME DISPLAY ====================
        function updateTime() {
            const now = new Date();
            document.getElementById('timeDisplay').textContent = now.toLocaleTimeString();
        }
        setInterval(updateTime, 1000);
        updateTime();

        // ==================== FETCH FUNCTIONS ====================
        async function fetchCounts() {
            try {
                const res = await fetch('/api/counts');
                const data = await res.json();
                document.getElementById('guppy').textContent = data.guppy || 0;
                document.getElementById('molly').textContent = data.molly || 0;
                document.getElementById('platy').textContent = data.platy || 0;
                document.getElementById('total').textContent = data.total_fish || 0;
            } catch(e) {
                console.error('Counts error:', e);
            }
        }
        
        async function fetchSensors() {
            try {
                const res = await fetch('/api/sensors');
                const data = await res.json();
                
                // Update sensor values
                document.getElementById('ph').textContent = (data.ph || 0).toFixed(2);
                document.getElementById('tds').textContent = data.tds || 0;
                document.getElementById('light').textContent = data.light || 0;
                document.getElementById('mq135').textContent = data.mq135 || 0;
                document.getElementById('dataCount').textContent = data.data_count || 0;
                
                // Float State
                const floatEl = document.getElementById('floatState');
                const floatState = data.float_state || 'UNKNOWN';
                floatEl.textContent = floatState;
                floatEl.className = 'sensor-value float-' + floatState.toLowerCase();
                
                // Last Update
                if (data.last_update) {
                    const date = new Date(data.last_update);
                    document.getElementById('lastUpdate').textContent = date.toLocaleTimeString();
                }
                
                // pH Status
                const ph = data.ph || 0;
                const phStatus = document.getElementById('phStatus');
                if (ph < 6.5) {
                    phStatus.textContent = 'ACIDIC';
                    phStatus.className = 'sensor-badge badge-acidic';
                } else if (ph > 7.5) {
                    phStatus.textContent = 'ALKALINE';
                    phStatus.className = 'sensor-badge badge-alkaline';
                } else {
                    phStatus.textContent = 'NEUTRAL';
                    phStatus.className = 'sensor-badge badge-neutral';
                }
                
                // MQ135 Status
                const mq135 = parseInt(data.mq135 || 0);
                const mq135Status = document.getElementById('mq135Status');
                if (mq135 < 80) {
                    mq135Status.textContent = 'GOOD';
                    mq135Status.className = 'sensor-badge badge-neutral';
                } else if (mq135 < 150) {
                    mq135Status.textContent = 'MODERATE';
                    mq135Status.className = 'sensor-badge badge-alkaline';
                } else {
                    mq135Status.textContent = 'POOR';
                    mq135Status.className = 'sensor-badge badge-acidic';
                }
                
                // Overall Status
                const dot = document.getElementById('statusDot');
                const label = document.getElementById('statusLabel');
                if (data.sensor_connected) {
                    dot.className = 'dot dot-online';
                    label.textContent = 'ESP32 Connected';
                    label.className = 'online';
                } else {
                    dot.className = 'dot dot-offline';
                    label.textContent = 'ESP32 Disconnected';
                    label.className = 'offline';
                }
                
            } catch(e) {
                console.error('Sensor error:', e);
                document.getElementById('statusDot').className = 'dot dot-offline';
                document.getElementById('statusLabel').textContent = 'Server Error';
                document.getElementById('statusLabel').className = 'offline';
            }
        }
        
        async function checkMLStatus() {
            try {
                const res = await fetch('/api/status');
                const data = await res.json();
                const indicator = document.getElementById('mlIndicator');
                const text = document.getElementById('mlStatusText');
                if (data.ml_server === 'Connected') {
                    indicator.className = 'ml-indicator';
                    text.textContent = 'Active';
                } else {
                    indicator.className = 'ml-indicator offline';
                    text.textContent = 'Offline';
                }
            } catch(e) {
                document.getElementById('mlIndicator').className = 'ml-indicator offline';
                document.getElementById('mlStatusText').textContent = 'Error';
            }
        }
        
        // ==================== INTERVALS ====================
        setInterval(fetchCounts, 1000);
        setInterval(fetchSensors, 1000);
        setInterval(checkMLStatus, 5000);
        
        // ==================== INITIAL FETCH ====================
        fetchCounts();
        fetchSensors();
        checkMLStatus();
    </script>
</body>
</html>
"""

if __name__ == '__main__':
    print("=" * 60)
    print("🐟 AQUAPONICS - RPi FRONTEND")
    print("=" * 60)
    print(f"📡 Windows ML Server: {WINDOWS_ML_SERVER}")
    print(f"📡 Windows Sensor Server: {WINDOWS_SENSOR_SERVER}")
    print("=" * 60)
    print("🌐 Starting on port 5001")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)