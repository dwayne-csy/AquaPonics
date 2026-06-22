# ml_backend.py - Pure Machine Learning Server (Windows)
from flask import Flask, jsonify, request
from flask_cors import CORS
import cv2
import numpy as np
from ultralytics import YOLO
import time
from collections import deque
import torch
import socket
import base64
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

# ==================== GPU SETUP ====================
print("=" * 60)
print("🤖 ML FISH DETECTION BACKEND (Windows)")
print("=" * 60)

if torch.cuda.is_available():
    print(f"✅ CUDA available - GPU: {torch.cuda.get_device_name(0)}")
    device = 'cuda'
else:
    print("⚠️ CUDA not available - using CPU")
    device = 'cpu'

# Load YOLO model
print("\n📦 Loading YOLO model...")
try:
    model = YOLO("AquaPonics.pt")
    model.to(device)
    print(f"✅ Model loaded on {device}!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

# Class names - FISH ONLY
class_names = {
    1: "Guppy",
    2: "Molly",
    3: "Platy"
}

# Global variables
last_counts = {"guppy": 0, "molly": 0, "platy": 0, "total_fish": 0}
fps_stats = deque(maxlen=30)

# ==================== IMAGE PROCESSING ====================
def enhance_image(frame):
    """Apply preprocessing for better detection"""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l_enhanced = clahe.apply(l)
    lab_enhanced = cv2.merge((l_enhanced, a, b))
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    enhanced = cv2.filter2D(enhanced, -1, kernel)
    
    return enhanced

def detect_fish(frame):
    """Detect fish in frame"""
    global last_counts, fps_stats
    
    start_time = time.time()
    
    enhanced = enhance_image(frame)
    small_frame = cv2.resize(enhanced, (480, 480))
    
    results = model(
        small_frame,
        conf=0.45,
        iou=0.4,
        verbose=False,
        imgsz=480,
        device=device,
        augment=True,
        agnostic_nms=True
    )
    
    counts = {"Guppy": 0, "Molly": 0, "Platy": 0}
    annotated = frame.copy()
    
    if results[0].boxes is not None and len(results[0].boxes) > 0:
        scale_x = frame.shape[1] / 480
        scale_y = frame.shape[0] / 480
        
        detections = []
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = class_names.get(class_id, "Unknown")
            
            if class_id == 0 or class_name == "Unknown":
                continue
            
            if conf >= 0.45:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1*scale_x), int(y1*scale_y), int(x2*scale_x), int(y2*scale_y)
                detections.append({
                    'class_name': class_name,
                    'conf': conf,
                    'bbox': (x1, y1, x2, y2)
                })
        
        if detections:
            detections.sort(key=lambda x: x['conf'], reverse=True)
            final_detections = []
            
            for det in detections:
                keep = True
                for kept in final_detections:
                    x1, y1, x2, y2 = det['bbox']
                    kx1, ky1, kx2, ky2 = kept['bbox']
                    
                    inter_x1 = max(x1, kx1)
                    inter_y1 = max(y1, ky1)
                    inter_x2 = min(x2, kx2)
                    inter_y2 = min(y2, ky2)
                    
                    if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                        area1 = (x2 - x1) * (y2 - y1)
                        area2 = (kx2 - kx1) * (ky2 - ky1)
                        iou = inter_area / (area1 + area2 - inter_area)
                        
                        if iou > 0.4:
                            keep = False
                            break
                
                if keep:
                    final_detections.append(det)
            
            colors = {
                "Guppy": (46, 204, 113),
                "Molly": (52, 152, 219),
                "Platy": (241, 196, 15)
            }
            
            for det in final_detections:
                class_name = det['class_name']
                conf = det['conf']
                x1, y1, x2, y2 = det['bbox']
                
                if class_name in counts:
                    counts[class_name] += 1
                
                color = colors.get(class_name, (46, 204, 113))
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
                
                label = f"{class_name} {conf:.2f}"
                (label_width, label_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                cv2.rectangle(annotated, 
                            (x1, y1 - label_height - 10), 
                            (x1 + label_width, y1), 
                            color, -1)
                cv2.putText(annotated, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    last_counts = {
        "guppy": counts["Guppy"],
        "molly": counts["Molly"],
        "platy": counts["Platy"],
        "total_fish": counts["Guppy"] + counts["Molly"] + counts["Platy"]
    }
    
    processing_time = time.time() - start_time
    fps_stats.append(1.0 / processing_time if processing_time > 0 else 0)
    current_fps = sum(fps_stats) / len(fps_stats) if fps_stats else 0
    
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (250, 100), (0, 0, 0), -1)
    annotated = cv2.addWeighted(overlay, 0.3, annotated, 0.7, 0)
    
    cv2.putText(annotated, f"FPS: {current_fps:.1f}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (46, 204, 113), 2)
    cv2.putText(annotated, f"Total Fish: {last_counts['total_fish']}", 
               (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(annotated, f"Guppy: {last_counts['guppy']} | Molly: {last_counts['molly']} | Platy: {last_counts['platy']}", 
               (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    cv2.putText(annotated, f"Device: {device.upper()}", 
               (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    return annotated

# ==================== API ROUTES ====================

@app.route('/api/detect', methods=['POST'])
def detect():
    """Detect fish in uploaded image"""
    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({'success': False, 'error': 'No image data'}), 400
        
        image_bytes = base64.b64decode(data['image'])
        image = Image.open(io.BytesIO(image_bytes))
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        annotated = detect_fish(frame)
        
        _, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
        annotated_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'success': True,
            'counts': last_counts,
            'annotated_image': annotated_base64
        })
        
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/counts')
def get_counts():
    return jsonify(last_counts)

@app.route('/api/status')
def get_status():
    return jsonify({
        "status": "running",
        "model": "AquaPonics.pt",
        "device": device.upper(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
        "fps": sum(fps_stats) / len(fps_stats) if fps_stats else 0,
        "classes": ["Guppy", "Molly", "Platy"],
        "accuracy_enhancements": [
            "CLAHE contrast enhancement",
            "Image sharpening",
            "Higher resolution (480x480)",
            "Test-time augmentation"
        ]
    })

@app.route('/')
def index():
    return jsonify({
        "service": "ML Fish Detection Backend",
        "version": "2.0",
        "endpoints": {
            "/api/detect": "POST - Send image for fish detection",
            "/api/counts": "GET - Get current fish counts",
            "/api/status": "GET - Server status"
        }
    })

if __name__ == '__main__':
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    
    print("\n" + "=" * 60)
    print("🚀 ML Fish Detection Server Running (Windows)")
    print("=" * 60)
    print(f"✅ Device: {device.upper()}")
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 60)
    print(f"🌐 Local: http://localhost:5000")
    print(f"🌐 Network: http://{local_ip}:5000")
    print("=" * 60)
    print("⚠️  Make sure AquaPonics.pt is in the current directory")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)