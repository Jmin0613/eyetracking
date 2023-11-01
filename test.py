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

warning_time = 0
warning_count = 0
DISPLAY_WARNING = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    landmarks = get_face_landmarks(frame)
    if landmarks:
        nose_tip = landmarks.part(30)
        mouth_left = landmarks.part(48)
        mouth_right = landmarks.part(54)

        cv2.line(frame, (nose_tip.x, nose_tip.y), (mouth_left.x, mouth_left.y), (255, 0, 0), 2)
        cv2.line(frame, (nose_tip.x, nose_tip.y), (mouth_right.x, mouth_right.y), (255, 0, 0), 2)

        ratio = get_distance(nose_tip, mouth_left) / get_distance(nose_tip, mouth_right)

        mouth_movement = get_mouth_movement(landmarks)

        # 경고 메시지를 띄우고 5초 동안 고개 회전 감지를 막음
        if DISPLAY_WARNING:
            if time.time() - warning_time < 5:
                cv2.putText(frame, "Warning: Head rotation detection", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                DISPLAY_WARNING = False
        else:
            # 미소를 지었는지 판별하고, 미소를 지지 않았을 경우에만 경고 메시지 출력 및 카운트
            if mouth_movement < MOUTH_THRESHOLD:
                if ratio > 1.2 or ratio < 0.8:  # 임의의 기준값
                    DISPLAY_WARNING = True
                    warning_time = time.time()
                    warning_count += 1

    cv2.putText(frame, f"Warning Count: {warning_count}", (frame.shape[1] - 160, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()