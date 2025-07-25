from deepface import DeepFace
import cv2
import datetime
import csv
import os
import numpy as np
import pickle
from collections import Counter
import time

def create_employee_database():
    """Create or load employee face database"""
    database_path = "employee_face_database.pkl"
    
    if os.path.exists(database_path):
        with open(database_path, 'rb') as f:
            return pickle.load(f)
    else:
        # Create empty database
        database = {}
        with open(database_path, 'wb') as f:
            pickle.dump(database, f)
        print("📁 Created new employee face database")
        return database

def save_employee_database(database):
    """Save employee face database"""
    with open("employee_face_database.pkl", 'wb') as f:
        pickle.dump(database, f)

def register_new_employee():
    """Register a new employee with their face"""
    print("\n🆕 EMPLOYEE REGISTRATION")
    print("="*30)
    
    employee_id = input("Enter Employee ID: ").strip()
    employee_name = input("Enter Employee Name: ").strip()
    
    if not employee_id or not employee_name:
        print("❌ Employee ID and Name are required!")
        return None
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open webcam")
        return None
    
    print(f"📷 Position your face in the camera frame for registration")
    print("Press 'c' to capture your face for registration")
    print("Press 'ESC' to cancel")
    
    face_encodings = []
    capture_count = 0
    required_captures = 3  # Capture multiple images for better accuracy
    
    while capture_count < required_captures:
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Display instructions
        cv2.putText(frame, f"Registration: {employee_name}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Captures: {capture_count}/{required_captures}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'c' to capture", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Employee Registration', frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('c'):
            try:
                # Get face embedding using DeepFace
                embedding = DeepFace.represent(frame, model_name='VGG-Face', enforce_detection=False)
                face_encodings.append(embedding[0]['embedding'])
                capture_count += 1
                print(f"✅ Captured face {capture_count}/{required_captures}")
                time.sleep(1)  # Pause between captures
                
            except Exception as e:
                print(f"❌ Failed to capture face: {e}")
        
        elif key == 27:  # ESC key
            print("❌ Registration cancelled")
            cap.release()
            cv2.destroyAllWindows()
            return None
    
    cap.release()
    cv2.destroyAllWindows()
    
    if len(face_encodings) == required_captures:
        # Calculate average embedding for better accuracy
        avg_embedding = np.mean(face_encodings, axis=0).tolist()
        
        employee_data = {
            'employee_id': employee_id,
            'name': employee_name,
            'face_embedding': avg_embedding,
            'registration_date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        print(f"✅ Successfully registered {employee_name} (ID: {employee_id})")
        return employee_data
    else:
        print("❌ Registration failed - insufficient face captures")
        return None

def recognize_employee_face(frame, employee_database, threshold=0.6):
    """Recognize employee from face using the database"""
    try:
        # Get face embedding from current frame
        current_embedding = DeepFace.represent(frame, model_name='VGG-Face', enforce_detection=False)
        current_vector = np.array(current_embedding[0]['embedding'])
        
        best_match = None
        best_distance = float('inf')
        
        # Compare with all registered employees
        for emp_id, emp_data in employee_database.items():
            stored_vector = np.array(emp_data['face_embedding'])
            
            # Calculate cosine distance
            distance = np.linalg.norm(current_vector - stored_vector)
            
            if distance < best_distance:
                best_distance = distance
                best_match = emp_data
        
        # Check if the best match is within threshold
        if best_match and best_distance < threshold:
            confidence = max(0, (threshold - best_distance) / threshold * 100)
            return {
                'employee_id': best_match['employee_id'],
                'name': best_match['name'],
                'confidence': confidence,
                'distance': best_distance
            }
        else:
            return None
            
    except Exception as e:
        print(f"❌ Face recognition error: {e}")
        return None

def get_employee_id_from_face(frame, employee_database):
    """Get employee ID through facial recognition"""
    print("🔍 Analyzing face for employee recognition...")
    
    recognition_result = recognize_employee_face(frame, employee_database)
    
    if recognition_result:
        print(f"✅ Employee recognized: {recognition_result['name']} (ID: {recognition_result['employee_id']})")
        print(f"🎯 Confidence: {recognition_result['confidence']:.1f}%")
        return recognition_result['name'], recognition_result['employee_id']
    else:
        print("❌ Employee not recognized in database")
        print("💡 You may need to register this employee first")
        return "Unknown Employee", "Unknown"

def analyze_facial_health(frame):
    """Comprehensive facial health analysis"""
    health_report = {
        'skin_analysis': {},
        'eye_analysis': {},
        'fatigue_level': 'Normal',
        'stress_indicators': [],
        'overall_wellness': 'Good',
        'recommendations': []
    }
    
    try:
        # Convert to different color spaces for analysis
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Face detection for focused analysis
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) > 0:
            # Analyze the largest detected face
            (x, y, w, h) = max(faces, key=lambda face: face[2] * face[3])
            face_roi = frame[y:y+h, x:x+w]
            face_gray = gray[y:y+h, x:x+w]
            face_hsv = hsv[y:y+h, x:x+w]
            face_lab = lab[y:y+h, x:x+w]
            
            # 1. SKIN ANALYSIS
            health_report['skin_analysis'] = analyze_skin_health(face_roi, face_hsv, face_lab)
            
            # 2. EYE ANALYSIS
            eyes = eye_cascade.detectMultiScale(face_gray)
            if len(eyes) >= 2:
                health_report['eye_analysis'] = analyze_eye_health(face_roi, eyes, face_gray)
            
            # 3. FATIGUE ANALYSIS
            health_report['fatigue_level'] = analyze_fatigue_level(face_roi, face_gray, eyes)
            
            # 4. STRESS INDICATORS
            health_report['stress_indicators'] = detect_stress_indicators(face_roi, face_gray)
            
            # 5. OVERALL WELLNESS ASSESSMENT
            health_report['overall_wellness'] = assess_overall_wellness(health_report)
            
            # 6. HEALTH RECOMMENDATIONS
            health_report['recommendations'] = generate_health_recommendations(health_report)
        
        else:
            health_report['error'] = 'No face detected for health analysis'
            
    except Exception as e:
        health_report['error'] = f'Health analysis error: {str(e)}'
    
    return health_report

def analyze_skin_health(face_roi, face_hsv, face_lab):
    """Analyze skin health indicators"""
    skin_analysis = {}
    
    try:
        # Skin tone analysis
        l_channel = face_lab[:, :, 0]
        avg_brightness = np.mean(l_channel)
        skin_analysis['brightness_level'] = 'Good' if 120 < avg_brightness < 180 else 'Needs Attention'
        
        # Skin uniformity (texture analysis)
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray_face, cv2.CV_64F).var()
        skin_analysis['skin_texture'] = 'Smooth' if laplacian_var < 500 else 'Rough'
        
        # Color analysis for health indicators
        b, g, r = cv2.split(face_roi)
        avg_red = np.mean(r)
        avg_green = np.mean(g)
        avg_blue = np.mean(b)
        
        # Redness indicator (potential inflammation/fatigue)
        redness_ratio = avg_red / (avg_green + avg_blue + 1)
        skin_analysis['redness_level'] = 'High' if redness_ratio > 0.6 else 'Normal'
        
        # Pallor detection
        pallor_indicator = (avg_red + avg_green + avg_blue) / 3
        skin_analysis['pallor_status'] = 'Pale' if pallor_indicator < 100 else 'Normal'
        
        # Hydration estimate (based on skin reflectance)
        skin_analysis['hydration_estimate'] = 'Good' if avg_brightness > 130 else 'Low'
        
    except Exception as e:
        skin_analysis['error'] = f'Skin analysis error: {str(e)}'
    
    return skin_analysis

def analyze_eye_health(face_roi, eyes, face_gray):
    """Analyze eye health and fatigue indicators"""
    eye_analysis = {}
    
    try:
        if len(eyes) >= 2:
            # Analyze eye regions
            eye_regions = []
            for (ex, ey, ew, eh) in eyes[:2]:  # Analyze first two eyes
                eye_region = face_gray[ey:ey+eh, ex:ex+ew]
                eye_regions.append(eye_region)
            
            # Eye brightness (indicator of alertness)
            avg_eye_brightness = np.mean([np.mean(eye) for eye in eye_regions])
            eye_analysis['eye_brightness'] = 'Bright' if avg_eye_brightness > 100 else 'Dim'
            
            # Eye opening assessment
            eye_openness = []
            for eye_region in eye_regions:
                # Simple eye openness based on variance in vertical direction
                vertical_profile = np.mean(eye_region, axis=1)
                eye_variance = np.var(vertical_profile)
                eye_openness.append(eye_variance)
            
            avg_openness = np.mean(eye_openness)
            eye_analysis['eye_openness'] = 'Wide Open' if avg_openness > 200 else 'Partially Closed' if avg_openness > 50 else 'Droopy'
            
            # Dark circles detection (simplified)
            under_eye_darkness = avg_eye_brightness < 80
            eye_analysis['dark_circles'] = 'Present' if under_eye_darkness else 'Minimal'
            
            # Eye symmetry
            if len(eye_openness) == 2:
                symmetry_diff = abs(eye_openness[0] - eye_openness[1])
                eye_analysis['eye_symmetry'] = 'Good' if symmetry_diff < 50 else 'Asymmetric'
        
    except Exception as e:
        eye_analysis['error'] = f'Eye analysis error: {str(e)}'
    
    return eye_analysis

def analyze_fatigue_level(face_roi, face_gray, eyes):
    """Analyze fatigue level from facial features"""
    fatigue_indicators = []
    
    try:
        # Eye drooping indicator
        if len(eyes) >= 2:
            eye_positions = [ey + eh/2 for (ex, ey, ew, eh) in eyes[:2]]
            if len(eye_positions) == 2:
                eye_level_diff = abs(eye_positions[0] - eye_positions[1])
                if eye_level_diff > 10:
                    fatigue_indicators.append('uneven_eyes')
        
        # Overall facial muscle tension
        edge_density = len(cv2.Canny(face_gray, 50, 150).nonzero()[0])
        face_area = face_gray.shape[0] * face_gray.shape[1]
        tension_ratio = edge_density / face_area
        
        if tension_ratio < 0.1:
            fatigue_indicators.append('low_muscle_tension')
        
        # Determine fatigue level
        if len(fatigue_indicators) >= 2:
            return 'High'
        elif len(fatigue_indicators) == 1:
            return 'Moderate'
        else:
            return 'Low'
            
    except Exception as e:
        return 'Unable to assess'

def detect_stress_indicators(face_roi, face_gray):
    """Detect potential stress indicators from facial features"""
    stress_indicators = []
    
    try:
        # Forehead tension (wrinkles)
        forehead_region = face_gray[:face_gray.shape[0]//3, :]
        forehead_edges = cv2.Canny(forehead_region, 50, 150)
        forehead_line_density = np.sum(forehead_edges) / (forehead_region.shape[0] * forehead_region.shape[1])
        
        if forehead_line_density > 0.02:
            stress_indicators.append('forehead_tension')
        
        # Jaw tension (simplified detection)
        jaw_region = face_gray[2*face_gray.shape[0]//3:, :]
        jaw_variance = np.var(jaw_region)
        
        if jaw_variance > 1000:
            stress_indicators.append('jaw_tension')
        
        # Overall facial asymmetry as stress indicator
        left_half = face_gray[:, :face_gray.shape[1]//2]
        right_half = cv2.flip(face_gray[:, face_gray.shape[1]//2:], 1)
        
        # Resize to match if needed
        min_width = min(left_half.shape[1], right_half.shape[1])
        left_half = left_half[:, :min_width]
        right_half = right_half[:, :min_width]
        
        asymmetry = np.mean(np.abs(left_half.astype(float) - right_half.astype(float)))
        
        if asymmetry > 20:
            stress_indicators.append('facial_asymmetry')
            
    except Exception as e:
        stress_indicators.append(f'analysis_error: {str(e)}')
    
    return stress_indicators

def assess_overall_wellness(health_report):
    """Assess overall wellness based on all indicators"""
    wellness_score = 100
    
    # Skin health impact
    skin = health_report.get('skin_analysis', {})
    if skin.get('redness_level') == 'High':
        wellness_score -= 10
    if skin.get('pallor_status') == 'Pale':
        wellness_score -= 15
    if skin.get('hydration_estimate') == 'Low':
        wellness_score -= 10
    
    # Eye health impact
    eye = health_report.get('eye_analysis', {})
    if eye.get('dark_circles') == 'Present':
        wellness_score -= 15
    if eye.get('eye_openness') == 'Droopy':
        wellness_score -= 20
    
    # Fatigue impact
    if health_report.get('fatigue_level') == 'High':
        wellness_score -= 25
    elif health_report.get('fatigue_level') == 'Moderate':
        wellness_score -= 15
    
    # Stress impact
    stress_count = len(health_report.get('stress_indicators', []))
    wellness_score -= stress_count * 10
    
    # Determine wellness category
    if wellness_score >= 80:
        return 'Excellent'
    elif wellness_score >= 60:
        return 'Good'
    elif wellness_score >= 40:
        return 'Fair'
    else:
        return 'Poor'

def generate_health_recommendations(health_report):
    """Generate personalized health recommendations"""
    recommendations = []
    
    # Skin-based recommendations
    skin = health_report.get('skin_analysis', {})
    if skin.get('hydration_estimate') == 'Low':
        recommendations.append('Increase water intake for better skin hydration')
    if skin.get('pallor_status') == 'Pale':
        recommendations.append('Consider iron-rich foods and vitamin D supplementation')
    if skin.get('redness_level') == 'High':
        recommendations.append('Reduce stress and consider anti-inflammatory foods')
    
    # Eye-based recommendations
    eye = health_report.get('eye_analysis', {})
    if eye.get('dark_circles') == 'Present':
        recommendations.append('Ensure 7-8 hours of quality sleep')
    if eye.get('eye_openness') == 'Droopy':
        recommendations.append('Take regular breaks from screen time')
    
    # Fatigue recommendations
    if health_report.get('fatigue_level') in ['High', 'Moderate']:
        recommendations.append('Consider rest and stress management techniques')
    
    # Stress recommendations
    if len(health_report.get('stress_indicators', [])) > 1:
        recommendations.append('Practice relaxation techniques like meditation or deep breathing')
    
    # General wellness
    if health_report.get('overall_wellness') in ['Fair', 'Poor']:
        recommendations.append('Consider consulting with a healthcare professional')
    
    return recommendations if recommendations else ['Maintain current healthy habits']

def create_health_attendance_log():
    """Create comprehensive attendance log with health data"""
    if not os.path.exists("health_attendance_log.csv"):
        with open("health_attendance_log.csv", "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                "Name", "Employee_ID", "Timestamp", "Emotion", "Overall_Wellness", "Fatigue_Level",
                "Skin_Health", "Eye_Health", "Stress_Level", "Health_Recommendations",
                "Skin_Brightness", "Skin_Hydration", "Eye_Openness", "Dark_Circles"
            ])

def mark_health_attendance(name, employee_id, emotion, health_report):
    """Save comprehensive health attendance to CSV"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Extract key health metrics
    skin = health_report.get('skin_analysis', {})
    eye = health_report.get('eye_analysis', {})
    
    health_data = [
        name,
        employee_id,
        timestamp,
        emotion,
        health_report.get('overall_wellness', 'Unknown'),
        health_report.get('fatigue_level', 'Unknown'),
        skin.get('brightness_level', 'Unknown'),
        eye.get('eye_brightness', 'Unknown'),
        len(health_report.get('stress_indicators', [])),
        '; '.join(health_report.get('recommendations', [])),
        skin.get('brightness_level', 'Unknown'),
        skin.get('hydration_estimate', 'Unknown'),
        eye.get('eye_openness', 'Unknown'),
        eye.get('dark_circles', 'Unknown')
    ]
    
    with open("health_attendance_log.csv", "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(health_data)
    
    print(f"✅ Health attendance marked for {name} (ID: {employee_id}) at {timestamp}")
    print(f"📊 Overall Wellness: {health_report.get('overall_wellness', 'Unknown')}")
    print(f"😴 Fatigue Level: {health_report.get('fatigue_level', 'Unknown')}")
    print(f"💡 Recommendations: {health_report.get('recommendations', [])}")

def display_health_summary(health_report):
    """Display health summary on console"""
    print("\n" + "="*50)
    print("🏥 HEALTH ANALYSIS SUMMARY")
    print("="*50)
    
    print(f"Overall Wellness: {health_report.get('overall_wellness', 'Unknown')}")
    print(f"Fatigue Level: {health_report.get('fatigue_level', 'Unknown')}")
    
    # Skin Analysis
    skin = health_report.get('skin_analysis', {})
    if skin:
        print(f"\n👤 Skin Health:")
        for key, value in skin.items():
            if key != 'error':
                print(f"  • {key.replace('_', ' ').title()}: {value}")
    
    # Eye Analysis
    eye = health_report.get('eye_analysis', {})
    if eye:
        print(f"\n👁️ Eye Health:")
        for key, value in eye.items():
            if key != 'error':
                print(f"  • {key.replace('_', ' ').title()}: {value}")
    
    # Stress Indicators
    stress = health_report.get('stress_indicators', [])
    if stress:
        print(f"\n⚠️ Stress Indicators: {', '.join(stress)}")
    
    # Recommendations
    recommendations = health_report.get('recommendations', [])
    if recommendations:
        print(f"\n💡 Health Recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
    
    print("="*50)

def display_menu():
    """Display main menu options"""
    print("\n" + "="*50)
    print("🏢 ENHANCED HEALTH & ATTENDANCE SYSTEM")
    print("="*50)
    print("1. Start Attendance System")
    print("2. Register New Employee")
    print("3. View Employee Database")
    print("4. Exit")
    print("="*50)

def view_employee_database(database):
    """View all registered employees"""
    print("\n" + "="*40)
    print("👥 REGISTERED EMPLOYEES")
    print("="*40)
    
    if not database:
        print("No employees registered yet.")
    else:
        for emp_id, emp_data in database.items():
            print(f"ID: {emp_data['employee_id']} | Name: {emp_data['name']} | Registered: {emp_data['registration_date']}")
    
    print("="*40)

def main():
    """Main function to run the enhanced attendance system"""
    # Load employee database
    employee_database = create_employee_database()
    
    # Create health attendance log
    create_health_attendance_log()
    
    while True:
        display_menu()
        choice = input("Enter your choice (1-4): ").strip()
        
        if choice == '1':
            # Start Attendance System
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                print("❌ Error: Could not open webcam")
                continue
            
            print("📷 Enhanced Health & Attendance System Started")
            print("Press 'q' to capture attendance and health analysis")
            print("Press 'ESC' to return to main menu")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Failed to capture frame")
                    continue
                
                # Display frame
                cv2.putText(frame, "Press 'q' for Health Analysis", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow('Enhanced Health & Attendance System', frame)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\n🧠 Starting comprehensive health analysis...")
                    
                    # Analyze emotion
                    try:
                        emotion_result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                        emotion = emotion_result[0]['dominant_emotion']
                    except Exception as e:
                        print(f"❌ Emotion analysis error: {e}")
                        emotion = "Unknown"
                    
                    # Comprehensive health analysis
                    health_report = analyze_facial_health(frame)
                    
                    # Get employee identification through facial recognition
                    employee_name, employee_id = get_employee_id_from_face(frame, employee_database)
                    
                    # Display health summary
                    display_health_summary(health_report)
                    
                    # Mark attendance with health data
                    mark_health_attendance(employee_name, employee_id, emotion, health_report)
                    
                    print(f"\n✅ Attendance completed for {employee_name} (ID: {employee_id})")
                    print("Press 'q' again for another person or ESC to return to menu")
                    
                elif key == 27:  # ESC key
                    break
            
            cap.release()
            cv2.destroyAllWindows()
            
        elif choice == '2':
            # Register New Employee
            new_employee = register_new_employee()
            if new_employee:
                employee_database[new_employee['employee_id']] = new_employee
                save_employee_database(employee_database)
                print(f"✅ Employee {new_employee['name']} added to database")
        
        elif choice == '3':
            # View Employee Database
            view_employee_database(employee_database)
        
        elif choice == '4':
            # Exit
            print("👋 Thank you for using the Enhanced Health & Attendance System!")
            break
        
        else:
            print("❌ Invalid choice. Please enter 1-4.")

if __name__ == "__main__":
    main()