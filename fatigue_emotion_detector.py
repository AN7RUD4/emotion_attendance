import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
from imutils import face_utils
import time
from deepface import DeepFace
import json
from collections import Counter
import os

# ---------------- CONFIG ---------------- #
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"  # update path if needed
EAR_THRESHOLD = 0.25
EAR_CONSEC_FRAMES = 20
WINDOW_SECONDS = 60   # for testing: set to 10
FPS = 30              # for testing: set to 10
RISK_THRESHOLD = 0.5
# ---------------------------------------- #

# Initialize DLib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

# EAR calculation
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Simple skin tone heuristic
def analyze_skin_tone(face_img):
    hsv = cv2.cvtColor(face_img, cv2.COLOR_RGB2HSV)
    mean_hsv = np.mean(hsv, axis=(0, 1))
    hue, sat, val = mean_hsv
    pallor_score = 1.0 if sat < 50 and val > 150 else 0.0
    yellowing_score = 1.0 if 15 < hue < 30 else 0.0
    return pallor_score, yellowing_score

# History appending
def append_to_history(entry, history_file="health_history.json"):
    try:
        # If file does not exist or is empty → initialize
        if not os.path.exists(history_file) or os.stat(history_file).st_size == 0:
            with open(history_file, "w") as f:
                json.dump({"entries": []}, f, indent=4)

        with open(history_file, "r+") as f:
            try:
                hist = json.load(f)
            except json.JSONDecodeError:
                # If corrupted → reset
                hist = {"entries": []}
            hist["entries"].append(entry)
            f.seek(0)
            f.truncate()
            json.dump(hist, f, indent=4)
    except Exception as e:
        print(f"Error writing history: {e}")

# Initialize JSONs
def initialize_json():
    initial_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "health_state": "Initializing",
        "avg_ear": 0.0,
        "drowsy_percent": 0.0,
        "fatigue_level": 0.0,
        "stress_level": 0.0,
        "dominant_emotion": "None",
        "pallor_score": 0.0,
        "yellowing_score": 0.0,
        "health_risk_score": 0.0,
        "explainability": "System initialized",
        "escalation": "Waiting for first analysis"
    }
    with open("health_state.json", "w") as f:
        json.dump(initial_data, f, indent=4)
    print("Initialized health_state.json")
    if not os.path.exists("health_history.json"):
        with open("health_history.json", "w") as f:
            json.dump({"entries": []}, f, indent=4)
        print("Initialized health_history.json")

# ---------------- MAIN LOOP ---------------- #
def main():
    initialize_json()

    WINDOW_FRAMES = WINDOW_SECONDS * FPS
    COUNTER = 0
    ALERT = False

    ear_values, emotions, pallor_scores, yellowing_scores = [], [], [], []
    start_time = time.time()
    frame_count = 0

    # daily dataset image
    os.makedirs("dataset", exist_ok=True)
    current_date = time.strftime("%Y-%m-%d")
    pic_path = f"dataset/{current_date}.jpg"
    saved_today = os.path.exists(pic_path)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break
            frame_count += 1

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)
            current_emotion, pallor_score, yellowing_score = "Not Detected", 0.0, 0.0

            if not faces:
                emotions.append("Not Detected")
                pallor_scores.append(0.0)
                yellowing_scores.append(0.0)
            else:
                if not saved_today:
                    cv2.imwrite(pic_path, frame)
                    saved_today = True
                    print(f"Saved daily image: {pic_path}")

            for face in faces:
                shape = predictor(gray, face)
                shape = face_utils.shape_to_np(shape)
                leftEye, rightEye = shape[42:48], shape[36:42]

                leftEAR, rightEAR = eye_aspect_ratio(leftEye), eye_aspect_ratio(rightEye)
                ear = (leftEAR + rightEAR) / 2.0
                ear_values.append(ear)

                cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 1)

                if ear < EAR_THRESHOLD:
                    COUNTER += 1
                    if COUNTER >= EAR_CONSEC_FRAMES:
                        ALERT = True
                        cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    COUNTER, ALERT = 0, False

                cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                try:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    x, y, w, h = face.left(), face.top(), face.width(), face.height()
                    face_img = rgb_frame[y:y+h, x:x+w]

                    result = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)
                    current_emotion = result[0]['dominant_emotion'].capitalize()
                    emotions.append(current_emotion)

                    pallor_score, yellowing_score = analyze_skin_tone(face_img)
                    pallor_scores.append(pallor_score)
                    yellowing_scores.append(yellowing_score)
                except Exception as e:
                    print(f"DeepFace or skin analysis error: {e}")
                    emotions.append("Not Detected")
                    pallor_scores.append(0.0)
                    yellowing_scores.append(0.0)

                cv2.putText(frame, f"Emotion: {current_emotion}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # ---------- One-minute analysis ---------- #
            if len(ear_values) >= WINDOW_FRAMES or time.time() - start_time >= WINDOW_SECONDS:
                avg_ear = np.mean(ear_values) if ear_values else 0.0
                drowsy_frames = sum(1 for e in ear_values if e < EAR_THRESHOLD)
                drowsy_percent = (drowsy_frames / len(ear_values)) * 100 if ear_values else 0.0
                valid_emotions = [e for e in emotions if e != "Not Detected"]
                dominant_emotion = Counter(valid_emotions).most_common(1)[0][0] if valid_emotions else "Neutral"
                avg_pallor = np.mean(pallor_scores) if pallor_scores else 0.0
                avg_yellowing = np.mean(yellowing_scores) if yellowing_scores else 0.0

                health_risk_score = (0.4 * drowsy_percent / 100 +
                                     0.3 * avg_pallor +
                                     0.3 * avg_yellowing)
                explainability = []
                if drowsy_percent > 50: explainability.append(f"High drowsiness ({drowsy_percent:.1f}%)")
                if avg_pallor > 0.5: explainability.append("Potential pallor detected")
                if avg_yellowing > 0.5: explainability.append("Potential yellowing detected")
                explain_text = "; ".join(explainability) if explainability else "No significant health signals"

                # Map to states
                if health_risk_score > RISK_THRESHOLD:
                    health_state, fatigue_level, stress_level, escalation = "Health Check Recommended", 0.6, 0.4, "Please consider a voluntary health check."
                elif drowsy_percent > 50:
                    health_state, fatigue_level, stress_level, escalation = "Drowsy", 0.8, 0.2, "Rest recommended."
                elif dominant_emotion in ['Sad', 'Fear', 'Angry']:
                    health_state, fatigue_level, stress_level, escalation = "Stressed", 0.3, 0.7, "Consider stress management."
                elif dominant_emotion == 'Happy':
                    health_state, fatigue_level, stress_level, escalation = "Happy", 0.1, 0.1, "Keep up the positive mood!"
                elif dominant_emotion == 'Neutral' and avg_ear < 0.3:
                    health_state, fatigue_level, stress_level, escalation = "High Fatigue", 0.6, 0.3, "Rest recommended."
                else:
                    health_state, fatigue_level, stress_level, escalation = "Normal", 0.2, 0.2, "No action needed."

                result = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "health_state": health_state,
                    "avg_ear": round(avg_ear, 2),
                    "drowsy_percent": round(drowsy_percent, 2),
                    "fatigue_level": round(fatigue_level, 2),
                    "stress_level": round(stress_level, 2),
                    "dominant_emotion": dominant_emotion,
                    "pallor_score": round(avg_pallor, 2),
                    "yellowing_score": round(avg_yellowing, 2),
                    "health_risk_score": round(health_risk_score, 2),
                    "explainability": explain_text,
                    "escalation": escalation
                }

                with open("health_state.json", "w") as f:
                    json.dump(result, f, indent=4)
                append_to_history(result)

                cv2.putText(frame, f"Health: {health_state}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, f"Action: {escalation}", (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                # reset AFTER writing
                ear_values, emotions, pallor_scores, yellowing_scores = [], [], [], []
                start_time = time.time()

            # display window
            cv2.imshow("Health Monitoring System", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

# ---------------- RUN ---------------- #
if __name__ == "__main__":
    main()
