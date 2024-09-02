from flask import Flask, render_template, request, redirect, url_for, Response
import numpy as np
import torch
from PIL import Image
import cv2
import os

app = Flask(__name__)

# Define the path to your custom YOLO model
model_path = os.path.join('models', 'best.pt')

# Load the custom YOLO model (set to CPU explicitly)
device = torch.device('cpu')
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path).to(device)

# Route for the Login Page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        # Implement authentication logic here
        if username == 'admin' and password == 'password':  # Example credentials
            return redirect(url_for('index'))
        else:
            return "Invalid credentials", 401
    return render_template('login.html')

# Route for the Home Page (Object Detection)
@app.route('/')
def index():
    return render_template('index.html')

# Route to Handle Image Upload and Object Detection
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))
    
    if file:
        # Save the uploaded file
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)

        # Perform object detection
        img = Image.open(file_path)
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).unsqueeze(0).float().to(device)
        results = model(img_tensor)

        # Save the detection results image
        results.save(save_dir='static/detections')  # Save the results in the static folder
        detection_image_path = os.path.join('static/detections', file.filename)
        
        # Render the results page with the detection image
        return render_template('result.html', image_url=detection_image_path)

# Route for the Webcam Page
@app.route('/webcam')
def webcam():
    return render_template('webcam.html')

# Function to generate frames from the webcam for live detection
def generate_frames():
    camera = cv2.VideoCapture(0)  # Open the default webcam

    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Perform object detection on the frame
            img_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float().to(device)
            results = model(img_tensor)

            # Render the detected objects on the frame
            results.render()
            
            # Convert frame to JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Yield the frame in byte format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    camera.release()

# Route to handle the video feed from the webcam
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
