import cv2
import face_recognition
import numpy as np
import os
import pickle
import time
import csv
from datetime import datetime

# Config
KNOWN_FACES_DIR = "imageAttendance"
VIDEO_PATH = "data/videoyt-loksabha.mp4"
ENCODINGS_FILE = "core/face_detection/encodings.pkl"
ATTENDANCE_FILE = "Attendance.csv"

known_face_encodings = []
known_face_names = []

# Load DNN face detection model (OpenCV SSD)


# Load or encode known faces


if os.path.exists(ENCODINGS_FILE):
    with open(ENCODINGS_FILE, "rb") as f:
        known_face_encodings, known_face_names = pickle.load(f)
    print("✅ Loaded known face encodings from cache")
else:
    print("🔄 Encoding known faces...")
    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            name = os.path.splitext(filename)[0]
            path = os.path.join(KNOWN_FACES_DIR, filename)
            image = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_face_encodings.append(encodings[0])
                known_face_names.append(name)
                print(f"✅ Encoded: {filename}")
            else:
                print(f"❌ No face found in {filename}, skipping.")
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump((known_face_encodings, known_face_names), f)
    print(f"✅ Encoding completed and saved to '{ENCODINGS_FILE}'")


# Load face detection model
print("Loading face detection models...")
try:
    modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
    configFile = "deploy.prototxt"
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    use_dnn = True
    print("Using DNN face detector")
except Exception as e:
    print(f"Warning: Could not load DNN model: {e}")
    print("Falling back to Haar Cascade detector")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    use_dnn = False

profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')


# Track attendance
attendance = {}

def mark_attendance(name):
    if not name:
        return

    if not os.path.isfile('Attendance.csv'):
        with open('Attendance.csv', 'w') as f:
            f.write("Name,Time,Date\n")

    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []

        for line in myDataList:
            entry = line.strip().split(',')
            if entry and len(entry) > 0:
                nameList.append(entry[0])

        now = datetime.now()
        dtString = now.strftime('%H:%M:%S')
        dateString = now.strftime('%Y-%m-%d')

        if name not in nameList:
            f.writelines(f'\n{name},{dtString},{dateString}')
            print(f"Marked attendance for: {name}")

# Start video
video_capture = cv2.VideoCapture(VIDEO_PATH)
if not video_capture.isOpened():
    print(f"❌ Error: Could not open video at {VIDEO_PATH}")
    exit()

fps = video_capture.get(cv2.CAP_PROP_FPS)
width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

frame_skip = 100
start_time = time.time()
frames_processed = 0
faces_detected = 0

cv2.namedWindow('Face Recognition with Attendance', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Face Recognition with Attendance', width, height)

print(f"Starting video processing: {width}x{height} at {fps:.2f} FPS")

frame_counter = 0

try:
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        if frame_counter % frame_skip == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
                name = "Unknown"

                if True in matches:
                    match_index = matches.index(True)
                    name = known_face_names[match_index]
                    mark_attendance(name)

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            elapsed_time = time.time() - start_time
            fps_text = f"FPS: {frames_processed / max(1, elapsed_time):.1f}"
            cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            progress = f"Frame: {frame_counter}/{total_frames} ({100*frame_counter/max(1,total_frames):.1f}%)"
            cv2.putText(frame, progress, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow('Face Recognition with Attendance', frame)
            frames_processed += 1

        frame_counter += 1
        key = cv2.waitKey(max(1, int(1000 / fps)))

        if key == 27:  # ESC
            break
        elif key == ord('s'):  # Skip 5 seconds
            frame_counter += int(fps * 5)
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_counter)

except Exception as e:
    print(f"Error during processing: {e}")
finally:
    video_capture.release()
    cv2.destroyAllWindows()

    # Save attendance to CSV
    with open("attendance.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Name", "Time"])
        for name, time_str in attendance.items():
            writer.writerow([name, time_str])

    print("\n--- Performance Summary ---")
    print(f"Total frames processed: {frames_processed}")
    print(f"Total faces detected: {faces_detected}")
    print(f"Average FPS: {frames_processed / max(1, time.time() - start_time):.2f}")