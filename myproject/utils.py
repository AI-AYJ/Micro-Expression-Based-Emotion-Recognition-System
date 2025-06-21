import torch
import torch.nn.functional as F
import cv2
from torchvision import transforms
from PIL import Image
import numpy as np

# OpenCV 얼굴 탐지 모델
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def crop_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    return frame[y:y+h, x:x+w]

def extract_frames(video_path, frame_skip=10):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_skip == 0:
            cropped = crop_face(frame)
            if cropped is not None:
                frames.append(cropped)
        frame_count += 1
    cap.release()
    return frames

def extract_optical_flow_frames(video_path, frame_skip=10, mag_thresh=1.0):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_buffer = []
    face_boxes = []  # 얼굴 좌표 저장

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_buffer.append(frame)

        # 처음 프레임에서만 얼굴 검출
        face = crop_face(frame)
        if face is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)
            if len(faces) > 0:
                face_boxes.append(faces[0])  # (x, y, w, h)
            else:
                face_boxes.append(None)
        else:
            face_boxes.append(None)

    cap.release()

    for i in range(0, len(frame_buffer) - frame_skip, frame_skip):
        img_on = frame_buffer[i]
        img_ap = frame_buffer[i + frame_skip]

        box = face_boxes[i]
        if box is None:
            continue
        x, y, w, h = box

        # 얼굴 영역 자르기
        face_on = img_on[y:y+h, x:x+w]
        face_ap = img_ap[y:y+h, x:x+w]

        g1 = cv2.cvtColor(face_on, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(face_ap, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            g1, g2, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)

        hsv = np.zeros_like(face_on)
        hsv[..., 1] = 255
        hsv[..., 0] = (ang / 2).astype(np.uint8)
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        hsv[..., 2][mag < mag_thresh] = 0
        flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        frames.append(flow_rgb)

    return frames




def ensemble_predict(model1, model2, x1, x2, weight1=0.5, weight2=0.5):
    with torch.no_grad():
        softmax1 = F.softmax(model1(x1), dim=1)
        softmax2 = F.softmax(model2(x2), dim=1)
        final_score = weight1 * softmax1 + weight2 * softmax2
        prediction = torch.argmax(final_score, dim=1).item()
    return prediction

def predict_single_image_ensemble(model1, model2, frame_apex, frame_flow, transform):
    try:
        # Apex에서 얼굴 좌표 검출
        gray = cv2.cvtColor(frame_apex, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        if len(faces) == 0:
            print("[DEBUG] ❌ Apex에서 얼굴 탐지 실패 - 예측 안함")
            return None

        x, y, w, h = faces[0]

        # 동일한 좌표로 두 프레임에서 얼굴 crop
        face_apex = frame_apex[y:y+h, x:x+w]
        face_flow = frame_flow[y:y+h, x:x+w]

        # BGR → PIL 이미지로 변환
        img1 = Image.fromarray(cv2.cvtColor(face_apex, cv2.COLOR_BGR2RGB))
        img2 = Image.fromarray(cv2.cvtColor(face_flow, cv2.COLOR_BGR2RGB))

        # 전처리 및 배치 차원 추가
        x1 = transform(img1).unsqueeze(0)
        x2 = transform(img2).unsqueeze(0)

        print("[DEBUG] ✔✔ Apex + Optical Flow 앙상블 예측 수행됨")
        return ensemble_predict(model1, model2, x1, x2)

    except Exception as e:
        print(f"[ERROR] 앙상블 예측 실패: {e}")
        return None


