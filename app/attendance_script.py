import cv2
import face_recognition
import numpy as np
import os
import pickle
import time
import csv
from datetime import datetime

# Config
KNOWN_FACES_DIR = "data/loksabha-img-mini"
VIDEO_PATH = "data/yt-lok2-mini.mp4"
ENCODINGS_FILE = "encodings.pkl"
ATTENDANCE_FILE = "Attendance.csv"

known_face_encodings = []
known_face_names = []

# Load or encode known faces
if os.path.exists(ENCODINGS_FILE):
    with open(ENCODINGS_FILE, "rb") as f:
        known_face_encodings, known_face_names = pickle.load(f)
    print("Loaded known face encodings from cache")
else:
    print(" Encoding known faces...")
    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            name = os.path.splitext(filename)[0]
            path = os.path.join(KNOWN_FACES_DIR, filename)
            image = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_face_encodings.append(encodings[0])
                known_face_names.append(name)
                print(f" Encoded: {filename}")
            else:
                print(f" No face found in {filename}, skipping.")
    
    # Save encodings to file
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump((known_face_encodings, known_face_names), f)
    print(f" Encoding completed and saved to '{ENCODINGS_FILE}'")

# Try to load DNN face detector models
print("Loading face detection models...")
use_dnn = False
try:
    # Download these files if you don't have them:
    # https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector/deploy.prototxt
    # https://github.com/opencv/opencv_3rdparty/blob/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel
    modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
    configFile = "deploy.prototxt"
    
    # Check if files exist before loading
    if os.path.exists(modelFile) and os.path.exists(configFile):
        net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
        use_dnn = True
        print("Using DNN face detector")
    else:
        raise FileNotFoundError("DNN model files not found")
except Exception as e:
    print(f"Warning: Could not load DNN model: {e}")
    print("Falling back to Haar Cascade detector")

# Load Haar Cascade detectors
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

# Function to mark attendance
# Function to mark attendance

def mark_attendance(name):
    if not name or name == "Unknown":
        return
    
    current_date = datetime.now().strftime('%Y-%m-%d')
    current_time = datetime.now().strftime('%H:%M:%S')
    
    # Create a lock file to prevent concurrent access issues
    lock_file = f"{ATTENDANCE_FILE}.lock"
    
    # Create the attendance file with headers if it doesn't exist
    if not os.path.isfile(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Time", "Date"])
    
    # Check if the person is already marked for today
    already_marked = False
    
    try:
        # First check if already marked without modifying the file
        with open(ATTENDANCE_FILE, 'r', newline='') as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip header
            
            for row in reader:
                if len(row) >= 3 and row[0] == name and row[2] == current_date:
                    already_marked = True
                    print(f"!! {name} already marked present today!")
                    break
        
        # If not already marked, add the new entry
        if not already_marked:
            # Use append mode to avoid reading the whole file
            with open(ATTENDANCE_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([name, current_time, current_date])
            print(f" Marked attendance for: {name}")
            
    except Exception as e:
        print(f"Error marking attendance: {e}")

def detect_faces(frame):
    """Detect faces in the frame using the selected method"""
    face_locations = []
    
    if use_dnn:
        # DNN detection (more accurate)
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), 
                                     [104, 117, 123], False, False)
        net.setInput(blob)
        detections = net.forward()
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > 0.6:  # Higher threshold for better accuracy
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")
                
                # Ensure coordinates are within frame boundaries
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)
                
                # Only add face if it has a meaningful size
                if x2 > x1 and y2 > y1 and (x2-x1)*(y2-y1) > 400:
                    # Convert to face_recognition format (top, right, bottom, left)
                    face_locations.append((y1, x2, y2, x1))
    else:
        # Haar Cascade detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        haar_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        for (x, y, w, h) in haar_faces:
            # Convert to face_recognition format (top, right, bottom, left)
            face_locations.append((y, x+w, y+h, x))
            
        # Try to detect profile faces if few faces were found
        if len(face_locations) < 2:
            # Try both original and flipped image to catch profiles facing both directions
            profile_faces = profile_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            for (x, y, w, h) in profile_faces:
                face_locations.append((y, x+w, y+h, x))
            
            # Try flipped image for profiles facing the other way
            flipped = cv2.flip(gray, 1)
            flipped_profile_faces = profile_cascade.detectMultiScale(flipped, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            # Convert flipped coordinates back to original image
            frame_width = gray.shape[1]
            for (x, y, w, h) in flipped_profile_faces:
                # Flip x-coordinate: new_x = width - (x + w)
                new_x = frame_width - (x + w)
                face_locations.append((y, new_x+w, y+h, new_x))
    
    return face_locations

# Main processing function
def process_video():
    print(" process_video() started!")
    print(" Opening video file...")
    # Start video
    video_capture = cv2.VideoCapture(VIDEO_PATH)
    if not video_capture.isOpened():
        print("[*] Reading frame...")


        print(f" Error: Could not open video at {VIDEO_PATH}")
        return

    fps = video_capture.get(cv2.CAP_PROP_FPS)
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:  # For webcam, total_frames is usually 0
        total_frames = float('inf')

    frame_skip = 5  # Process every 3rd frame for better performance
    start_time = time.time()
    frames_processed = 0
    faces_detected = 0
    
    # Recognition settings
    recognition_interval = 10 # Process face recognition every 30 frames
    recognition_counter = 0
    
    # Create window
    cv2.namedWindow('Face Recognition with Attendance', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Face Recognition with Attendance', width, height)

    print(f"Starting video processing: {width}x{height} at {fps:.2f} FPS")

    try:
        frame_counter = 0
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
                
            # Process every nth frame for detection
            if frame_counter % frame_skip == 0:
                # Always show current frame
                display_frame = frame.copy()
                
                # Face recognition happens at a different interval
                do_recognition = (recognition_counter % recognition_interval == 0)
                
                # Detect faces
                face_locations = detect_faces(frame)
                faces_detected += len(face_locations)
                
                # Process faces for recognition
                if face_locations and do_recognition and known_face_encodings:
                    # Get face encodings for detected faces
                    face_encodings = face_recognition.face_encodings(frame, face_locations)
                    
                    # Match each face
                    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                        # Try to match against known faces
                        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
                        name = "Unknown"
                        
                        # If match found, use the name of the first match
                        if True in matches:
                            match_index = matches.index(True)
                            name = known_face_names[match_index]
                            mark_attendance(name)
                        
                        # Draw on display frame
                        cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                        cv2.putText(display_frame, name, (left, top - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                elif face_locations:
                    # Just draw boxes without recognition
                    for (top, right, bottom, left) in face_locations:
                        cv2.rectangle(display_frame, (left, top), (right, bottom), (255, 0, 0), 2)
                
                # Display performance metrics
                elapsed_time = time.time() - start_time
                fps_text = f"FPS: {frames_processed / max(1, elapsed_time):.1f}"
                cv2.putText(display_frame, fps_text, (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # For video files, show progress
                if total_frames < float('inf'):
                    progress = f"Frame: {frame_counter}/{total_frames} ({100*frame_counter/total_frames:.1f}%)"
                    cv2.putText(display_frame, progress, (10, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Show the frame
                cv2.imshow('Face Recognition with Attendance', display_frame)
                frames_processed += 1
                recognition_counter += 1
            
            frame_counter += 1
            
            # Handle keyboard input
            key = cv2.waitKey(1)
            if key == 27:  # ESC
                break
            elif key == ord('s') and total_frames < float('inf'):  # Skip 5 seconds (for video files)
                frame_counter += int(fps * 5)
                video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_counter)
                
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        video_capture.release()
        cv2.destroyAllWindows()
        
        print("\n--- Performance Summary ---")
        print(f"Total frames processed: {frames_processed}")
        print(f"Total faces detected: {faces_detected}")
        print(f"Average FPS: {frames_processed / max(1, time.time() - start_time):.2f}")

    print(" Finished processing video.")


if __name__ == "__main__":
    process_video()


print(" attendance_script.py is being run directly")

process_video()