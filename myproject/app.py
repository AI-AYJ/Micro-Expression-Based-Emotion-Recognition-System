from flask import Flask, render_template, request, Response, jsonify
import os
import torch
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image
from model import SimpleCNN
from utils import extract_frames, extract_optical_flow_frames, predict_single_image_ensemble
from time import time

app = Flask(__name__)
app.secret_key = 'your-secret-key'
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

label_map = {
    0: "happiness",
    1: "disgust",
    2: "repression",
    3: "sadness",
    4: "surprise"
}
emoji_map = {
    0: "😄", 1: "🤢", 2: "😶", 3: "😢", 4: "😲"
}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

model1 = SimpleCNN(num_classes=5)
model2 = SimpleCNN(num_classes=5)
model1.load_state_dict(torch.load('SimpleCNN_model.pth', map_location='cpu'))
model2.load_state_dict(torch.load('optical_model.pth', map_location='cpu'))
model1.eval()
model2.eval()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    if file:
        filename = file.filename
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        apex_frames = extract_frames(filepath, frame_skip=10)
        flow_frames = extract_optical_flow_frames(filepath, frame_skip=10)

        result_log = []
        last_pred = None
        last_time = 0
        current_time = 0
        micro_count = 0
        total_changes = 0

        for f1, f2 in zip(apex_frames, flow_frames):
            pred = predict_single_image_ensemble(model1, model2, f1, f2, transform)
            if pred is None:
                current_pred = "얼굴을 비추세요 ❌"
            else:
                current_pred = f"{label_map[pred]} {emoji_map[pred]}"

            if current_pred != last_pred:
                if last_pred and last_pred != "얼굴을 비추세요 ❌" and current_pred != "얼굴을 비추세요 ❌":
                    duration = round(current_time - last_time, 1)
                    result_log.append(f"{last_pred} ({duration}초)")
                    total_changes += 1
                    if duration <= 0.4:
                        micro_count += 1

                if current_pred != "얼굴을 비추세요 ❌":
                    last_pred = current_pred
                    last_time = current_time

            current_time += 0.4

        stats = {
            "log": result_log,
            "micro_count": micro_count,
            "total_changes": total_changes,
            "percent": round((micro_count / total_changes) * 100, 1) if total_changes else 0.0
        }

        preview_path = None
        if apex_frames:
            preview_frame = apex_frames[0]
            preview_path = "preview.jpg"
            Image.fromarray(cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB)).save(os.path.join("static", preview_path))

        video_save_path = os.path.join("static", filename)
        file.seek(0)
        with open(video_save_path, 'wb') as f:
            f.write(file.read())

        return render_template('result.html', label="감정 분석 결과", emoji="", preview_image=preview_path, video_file=filename, stats=stats)

    return render_template('index.html', result="Upload failed")

camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

predicted_expression = "..."
last_pred = None
last_time = time()
expression_log = []
microexpression_flag = False
micro_count = 0
total_expression_changes = 0
last_infer_time = 0

@app.route('/webcam', methods=['GET', 'POST'])
def webcam():
    global expression_log, micro_count, total_expression_changes
    expression_log = []
    micro_count = 0
    total_expression_changes = 0
    return render_template('webcam.html', expression=predicted_expression)

@app.route('/video_feed')
def video_feed():
    def gen():
        global predicted_expression, last_pred, last_time, expression_log
        global microexpression_flag, micro_count, total_expression_changes, last_infer_time

        ret, prev = camera.read()
        if not ret:
            return
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        frame_count = 0

        while True:
            success, frame = camera.read()
            if not success:
                break
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            now = time()

            if now - last_infer_time > 0.4:
                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                hsv = np.zeros_like(frame)
                hsv[..., 1] = 255
                hsv[..., 0] = (ang / 2).astype(np.uint8)
                norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                hsv[..., 2] = norm.astype(np.uint8)

                # 작은 움직임 제거 (이 부분이 핵심!)
                hsv[..., 2][mag < 1.0] = 0

                flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

                # 👉 Optical Flow 이미지 디버깅 창 띄우기
                cv2.imshow("Optical Flow Debug", flow_rgb)
                cv2.waitKey(1)

                pred = predict_single_image_ensemble(model1, model2, frame, flow_rgb, transform)
                if pred is not None:
                    current_pred = f"{label_map[pred]} {emoji_map[pred]}"
                else:
                    current_pred = "얼굴을 비추세요 ❌"

                if current_pred != last_pred:
                    if last_pred and last_pred != "얼굴을 비추세요 ❌" and current_pred != "얼굴을 비추세요 ❌":
                        duration = round(now - last_time, 1)
                        expression_log.append(f"{last_pred} ({duration}초)")
                        total_expression_changes += 1
                        if duration <= 0.4:
                            micro_count += 1
                            microexpression_flag = True
                        else:
                            microexpression_flag = False

                    if current_pred != "얼굴을 비추세요 ❌":
                        last_pred = current_pred
                        last_time = now
                predicted_expression = current_pred
                last_infer_time = now

            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/current_expression')
def current_expression():
    return jsonify(expression=predicted_expression)

@app.route('/expression_log')
def get_log():
    return jsonify(log=expression_log[-10:][::-1])

@app.route('/microexpression')
def microexpression():
    return jsonify(micro=microexpression_flag)

@app.route('/micro_stats')
def micro_stats():
    percent = round((micro_count / total_expression_changes) * 100, 1) if total_expression_changes else 0.0
    return jsonify(count=micro_count, percent=percent, total=total_expression_changes)

if __name__ == '__main__':
    app.run(debug=True)
