from flask import Flask, render_template, Response
import cv2
import dlib
import numpy as np

app = Flask(__name__)

# dlib 얼굴 인식 모델 및 랜드마크 모델 로드
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def get_face_landmarks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        return None

    landmarks = predictor(gray, faces[0])
    return landmarks

def generate_frames():
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

            ratio = get_distance(nose_tip, mouth_left) / get_distance(nose_tip, mouth_right)

            mouth_movement = get_mouth_movement(landmarks)

            if DISPLAY_WARNING:
                if time.time() - warning_time < 5:
                    cv2.putText(frame, "Warning: Head rotation detection", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    DISPLAY_WARNING = False
            else:
                if mouth_movement < MOUTH_THRESHOLD:
                    if ratio > 1.2 or ratio < 0.8:
                        DISPLAY_WARNING = True
                        warning_time = time.time()
                        warning_count += 1

        cv2.putText(frame, f"Warning Count: {warning_count}", (frame.shape[1] - 160, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def get_distance(p1, p2):
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def get_mouth_movement(landmarks):
    upper_lip = landmarks.part(51)
    lower_lip = landmarks.part(57)
    distance = get_distance(upper_lip, lower_lip)
    return distance

@app.route('/')
def testPage():
    return render_template('testPage.html')

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)