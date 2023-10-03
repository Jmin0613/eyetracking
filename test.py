import cv2
import dlib
import numpy as np
import time

# dlib의 얼굴 인식 모델 및 랜드마크 모델 불러오기
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def get_face_landmarks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        return None

    landmarks = predictor(gray, faces[0])
    return landmarks

def get_distance(p1, p2):
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def get_mouth_movement(landmarks):
    # 입술 상단과 하단의 거리를 계산
    upper_lip = landmarks.part(51)
    lower_lip = landmarks.part(57)
    distance = get_distance(upper_lip, lower_lip)
    return distance

cap = cv2.VideoCapture(0)
MOUTH_THRESHOLD = 20  # 입술 움직임에 대한 임의의 임계값
warning_count = 0  # 경고 횟수 카운트 변수
last_warning_time = 0  # 마지막 경고 시간

# 경고 누적 제한 시간 (초)
WARNING_COOLDOWN_TIME = 5

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()

    # 경고 누적 제한 시간 내에는 경고를 누적하지 않음
    if current_time - last_warning_time >= WARNING_COOLDOWN_TIME:
        landmarks = get_face_landmarks(frame)
        if landmarks:
            nose_tip = landmarks.part(30)
            mouth_left = landmarks.part(48)
            mouth_right = landmarks.part(54)

            cv2.line(frame, (nose_tip.x, nose_tip.y), (mouth_left.x, mouth_left.y), (255, 0, 0), 2)
            cv2.line(frame, (nose_tip.x, nose_tip.y), (mouth_right.x, mouth_right.y), (255, 0, 0), 2)

            ratio = get_distance(nose_tip, mouth_left) / get_distance(nose_tip, mouth_right)

            mouth_movement = get_mouth_movement(landmarks)

            # 미소를 지었는지 판별하고, 미소를 지지 않았을 경우에만 경고 메시지 출력
            if mouth_movement < MOUTH_THRESHOLD:
                if ratio > 1.2 or ratio < 0.8:  # 임의의 기준값
                    warning_count += 1
                    last_warning_time = current_time
                    cv2.putText(frame, f"Warning: Head rotation detection", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 오른쪽 하단에 경고 횟수 표시
    cv2.putText(frame, f"Warnings: {warning_count}", (frame.shape[1] - 200, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()