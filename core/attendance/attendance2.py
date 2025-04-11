import cv2
import numpy as np
import time
import face_recognition
import os
import csv
from datetime import datetime

# ✅ Set your known face image folder path here
KNOWN_FACES_DIR = "loksabha-img"  # <-- CHANGE THIS TO YOUR IMAGE FOLDER

# Load known faces
known_face_encodings = []
known_face_names = []

print(f"Loading known faces from '{KNOWN_FACES_DIR}'...")
for file_name in os.listdir(KNOWN_FACES_DIR):
    if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(KNOWN_FACES_DIR, file_name)
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_face_encodings.append(encodings[0])
            known_face_names.append(os.path.splitext(file_name)[0])

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

# Attendance tracking
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

# Load video
video_path = './videos/videoyt-lokshab.mp4'
video_capture = cv2.VideoCapture(video_path)

if not video_capture.isOpened():
    print(f"Error: Could not open video at {video_path}")
    exit()

fps = video_capture.get(cv2.CAP_PROP_FPS)
width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

frame_skip = 10
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