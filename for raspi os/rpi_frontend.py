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
                temp = current_sensors.get('temperature', 0)
                ph_status = current_sensors.get('ph_status', 'NEUTRAL')
                float_state = current_sensors.get('float_state', 'UNKNOWN')
                data_count = current_sensors.get('data_count', 0)
                
                cv2.putText(annotated_frame, f"ML: {ml_server_status} | Sensors: {sensor_server_status}", (10, 110), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 200, 255), 1)
                cv2.putText(annotated_frame, f"pH: {ph:.2f} ({ph_status})  TDS: {tds}ppm", (10, 130), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 100), 1)
                cv2.putText(annotated_frame, f"Light: {light}lx  MQ135: {mq135}  Temp: {temp:.1f}C", (10, 150), 
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
<html>
<head>
    <title>AquaPonics - Fish Detection</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            background: #0a2f2a;
            font-family: Arial, sans-serif;
            color: #fff;
            padding: 20px;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        h1 { color: #2ecc71; text-align: center; margin-bottom: 20px; }
        .dashboard { display: flex; gap: 20px; flex-wrap: wrap; }
        .panel {
            background: rgba(0,0,0,0.5);
            border-radius: 15px;
            padding: 20px;
            border: 1px solid rgba(46,204,113,0.3);
            flex: 1;
            min-width: 200px;
        }
        .panel-wide { flex: 3; min-width: 500px; }
        .panel-title {
            color: #2ecc71;
            border-bottom: 2px solid #2ecc71;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }
        .camera-container {
            background: #000;
            border-radius: 10px;
            overflow: hidden;
        }
        .camera-container img { width: 100%; display: block; }
        .fish-item {
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid rgba(46,204,113,0.2);
        }
        .fish-count {
            font-size: 1.5rem;
            color: #2ecc71;
        }
        .total-fish {
            margin-top: 15px;
            padding: 15px;
            background: #2ecc71;
            border-radius: 12px;
            text-align: center;
        }
        .total-fish .fish-count {
            font-size: 2.5rem;
            color: #fff;
        }
        .sensor-item {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid rgba(46,204,113,0.15);
        }
        .sensor-value { color: #2ecc71; font-weight: bold; }
        .status-dot {
            display: inline-block;
            width: 10px;
            height: 10px;
            background: #2ecc71;
            border-radius: 50%;
            animation: pulse 1s infinite;
            margin-right: 8px;
        }
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        .status-disconnected .status-dot { background: #e74c3c; }
        .ph-neutral { color: #2ecc71; }
        .ph-acidic { color: #e74c3c; }
        .ph-alkaline { color: #f39c12; }
        .status-text {
            font-size: 0.8rem;
            padding: 2px 8px;
            border-radius: 10px;
        }
        .status-connected { color: #2ecc71; }
        .status-disconnected { color: #e74c3c; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🐟 AquaPonics</h1>
        <p style="text-align:center;color:#888;margin-bottom:20px;">
            Windows ML: {{ windows_ip }}:5000 | Windows Sensors: {{ windows_ip }}:5001
            <span id="statusText" class="status-text status-connected">● Connected</span>
        </p>
        
        <div class="dashboard">
            <div class="panel">
                <div class="panel-title">🐠 Fish Counter</div>
                <div class="fish-item">
                    <span>🐠 Guppy</span>
                    <span class="fish-count" id="guppy">0</span>
                </div>
                <div class="fish-item">
                    <span>🐟 Molly</span>
                    <span class="fish-count" id="molly">0</span>
                </div>
                <div class="fish-item">
                    <span>🐡 Platy</span>
                    <span class="fish-count" id="platy">0</span>
                </div>
                <div class="total-fish">
                    <div>🐟 TOTAL FISH</div>
                    <div class="fish-count" id="total">0</div>
                </div>
            </div>
            
            <div class="panel panel-wide">
                <div class="panel-title">📹 Live Feed</div>
                <div class="camera-container">
                    <img id="cameraFeed" src="/stream.mjpg" alt="Live Feed">
                </div>
            </div>
            
            <div class="panel">
                <div class="panel-title">💧 Sensors (ESP32)</div>
                <div class="sensor-item">
                    <span>📏 pH</span>
                    <span class="sensor-value"><span id="ph">0.00</span> <span id="phStatus">NEUTRAL</span></span>
                </div>
                <div class="sensor-item">
                    <span>💧 TDS</span>
                    <span class="sensor-value"><span id="tds">0</span> ppm</span>
                </div>
                <div class="sensor-item">
                    <span>🌡️ Temp</span>
                    <span class="sensor-value"><span id="temp">0.0</span> °C</span>
                </div>
                <div class="sensor-item">
                    <span>💡 Light</span>
                    <span class="sensor-value"><span id="light">0</span> lux</span>
                </div>
                <div class="sensor-item" style="background:rgba(46,204,113,0.05);padding:10px;border-radius:5px;">
                    <span>💨 MQ135</span>
                    <span class="sensor-value"><span id="mq135">0</span> ADC</span>
                </div>
                <div class="sensor-item">
                    <span>💧 Float</span>
                    <span class="sensor-value" id="float">UNKNOWN</span>
                </div>
                <div class="sensor-item">
                    <span>📊 Data</span>
                    <span class="sensor-value" id="dataCount">0</span>
                </div>
                <div class="sensor-item">
                    <span>🕐 Update</span>
                    <span class="sensor-value" id="lastUpdate">Never</span>
                </div>
            </div>
        </div>
    </div>

    <script>
        async function fetchCounts() {
            try {
                const res = await fetch('/api/counts');
                const data = await res.json();
                document.getElementById('guppy').textContent = data.guppy || 0;
                document.getElementById('molly').textContent = data.molly || 0;
                document.getElementById('platy').textContent = data.platy || 0;
                document.getElementById('total').textContent = data.total_fish || 0;
            } catch(e) { console.error(e); }
        }
        
        async function fetchSensors() {
            try {
                const res = await fetch('/api/sensors');
                const data = await res.json();
                
                document.getElementById('ph').textContent = (data.ph || 0).toFixed(2);
                document.getElementById('tds').textContent = data.tds || 0;
                document.getElementById('temp').textContent = (data.temperature || 0).toFixed(1);
                document.getElementById('light').textContent = data.light || 0;
                document.getElementById('mq135').textContent = data.mq135 || 0;
                document.getElementById('float').textContent = data.float_state || 'UNKNOWN';
                document.getElementById('dataCount').textContent = data.data_count || 0;
                
                // Last update time
                if (data.last_update) {
                    const date = new Date(data.last_update);
                    document.getElementById('lastUpdate').textContent = date.toLocaleTimeString();
                }
                
                // pH Status
                const status = document.getElementById('phStatus');
                const ph = data.ph || 0;
                if (ph < 6.5) { 
                    status.textContent = 'ACIDIC'; 
                    status.style.color = '#e74c3c'; 
                } else if (ph > 7.5) { 
                    status.textContent = 'ALKALINE'; 
                    status.style.color = '#f39c12'; 
                } else { 
                    status.textContent = 'NEUTRAL'; 
                    status.style.color = '#2ecc71'; 
                }
                
                // Update status text
                const statusText = document.getElementById('statusText');
                if (data.sensor_connected) {
                    statusText.textContent = '● ESP32 Connected';
                    statusText.className = 'status-text status-connected';
                } else {
                    statusText.textContent = '● ESP32 Disconnected';
                    statusText.className = 'status-text status-disconnected';
                }
            } catch(e) { 
                console.error(e);
                document.getElementById('statusText').textContent = '● Server Error';
                document.getElementById('statusText').className = 'status-text status-disconnected';
            }
        }
        
        setInterval(fetchCounts, 1000);
        setInterval(fetchSensors, 1000);
        fetchCounts();
        fetchSensors();
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