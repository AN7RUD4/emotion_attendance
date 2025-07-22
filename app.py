from deepface import DeepFace
import cv2
import datetime
import csv
import os
import speech_recognition as sr

def get_employee_id_from_voice():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎤 Please say your employee ID or name...")
        audio = recognizer.listen(source)

        try:
            text = recognizer.recognize_google(audio)
            print(f"🔍 You said: {text}")
            return text.strip()
        except sr.UnknownValueError:
            print("❌ Could not understand audio")
        except sr.RequestError as e:
            print(f"⚠️ Request error: {e}")
    return "Unknown"


# Create attendance log if not exists
if not os.path.exists("attendance_log.csv"):
    with open("attendance_log.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Name", "Timestamp", "Emotion"])

# Simulate face recognition (replace with real logic later)
def identify_employee():
    return get_employee_id_from_voice()

# Save attendance to CSV
def mark_attendance(name, emotion):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("attendance_log.csv", "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([name, timestamp, emotion])
    print(f"✅ Attendance marked for {name} at {timestamp} with emotion: {emotion}")

# Initialize webcam
cap = cv2.VideoCapture(0)

print("📷 Press 'q' to capture attendance")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    cv2.imshow('Emotion Attendance System', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Analyze the frame for emotion
print("🧠 Analyzing emotion...")
result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
emotion = result[0]['dominant_emotion']

# Identify employee and log attendance
employee_name = identify_employee()
mark_attendance(employee_name, emotion)

# Clean up
cap.release()
cv2.destroyAllWindows()