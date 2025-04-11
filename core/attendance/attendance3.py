import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import time

# Path for face image database
path = 'imageAttendance'
images = []
classNames = []

# Create the directory if it doesn't exist
os.makedirs(path, exist_ok=True)

# Check if directory is empty
myList = os.listdir(path)
if len(myList) == 0:
    print(f"Warning: No images found in {path}. Please add reference images.")
else:
    print(f"Found {len(myList)} images in database")

    # Load images and class names
    for cl in myList:
        if cl.lower().endswith(('.png', '.jpg', '.jpeg')):
            currImg = cv2.imread(f'{path}/{cl}')
            if currImg is not None:
                images.append(currImg)
                classNames.append(os.path.splitext(cl)[0])
                print(f"Loaded: {cl}")
            else:
                print(f"Could not load image: {cl}")

print(f"Successfully loaded {len(classNames)} faces")

# Load DNN face detector model
modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
configFile = "deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

def findEncodings(images):
    encodeList = []
    validNames = []

    for i, img in enumerate(images):
        print(f"Processing image {i+1}/{len(images)}")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(img_rgb)
        if not face_locations:
            print(f"No face detected in image for {classNames[i]}. Please use a clearer image.")
            continue

        try:
            encoding = face_recognition.face_encodings(img_rgb, face_locations)[0]
            encodeList.append(encoding)
            validNames.append(classNames[i])
            print(f"Successfully encoded: {classNames[i]}")
        except IndexError:
            print(f"Could not encode face for {classNames[i]}")

    return encodeList, validNames

def markAttendance(name):
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

if images:
    print("Starting face encoding process...")
    encodeListKnown, valid_names = findEncodings(images)
    print(f"Encoding Complete! Found {len(encodeListKnown)} valid face encodings")

    if len(encodeListKnown) == 0:
        print("No valid face encodings found. Please check your reference images.")
        exit()
else:
    print("No images to process. Please add images to the imageAttendance folder.")
    exit()

video_path = 'bright_output.mp4'
print(f"Opening video: {video_path}")

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video {video_path}")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video dimensions: {frame_width}x{frame_height}, FPS: {fps}, Total frames: {total_frames}")

frame_count = 0
process_time_total = 0
process_count = 0
attendance_record = {}
skip_frames = 2

def progress_bar(current, total, bar_length=50):
    fraction = current / total
    arrow = int(fraction * bar_length) * '█'
    padding = (bar_length - len(arrow)) * '░'
    return f"Progress: [{arrow}{padding}] {int(fraction * 100)}%"

print("Starting video processing. Press 'q' to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("End of video or failed to read frame")
        break

    frame_count += 1

    if frame_count % 100 == 0 or frame_count == 1:
        print(f"{progress_bar(frame_count, total_frames)} - Frame {frame_count}/{total_frames}")

    if frame_count % skip_frames != 0:
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    start_time = time.time()

    scale = 0.5 if frame_width <= 640 else 0.25
    small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    (h, w) = rgb_small_frame.shape[:2]
    blob = cv2.dnn.blobFromImage(rgb_small_frame, 1.0, (300, 300), [104.0, 177.0, 123.0], False, False)
    net.setInput(blob)
    detections = net.forward()

    face_locations = []
    confidence_threshold = 0.5
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            top = max(0, y1)
            right = min(w, x2)
            bottom = min(h, y2)
            left = max(0, x1)
            face_locations.append((top, right, bottom, left))

    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    detected_names = []

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        top = int(top / scale)
        right = int(right / scale)
        bottom = int(bottom / scale)
        left = int(left / scale)

        matches = face_recognition.compare_faces(encodeListKnown, face_encoding, tolerance=0.6)
        name = "Unknown"
        confidence = 0

        face_distances = face_recognition.face_distance(encodeListKnown, face_encoding)

        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            confidence = 1 - face_distances[best_match_index]

            if matches[best_match_index] and confidence > 0.4:
                name = valid_names[best_match_index].upper()
                detected_names.append(name)

                current_time = time.time()
                cooldown_seconds = 30

                if name not in attendance_record or current_time - attendance_record[name] > cooldown_seconds:
                    markAttendance(name)
                    attendance_record[name] = current_time

        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        label = f"{name} ({confidence:.2f})" if name != "Unknown" else "Unknown"
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        cv2.putText(frame, label, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

    process_time = time.time() - start_time
    process_time_total += process_time
    process_count += 1
    avg_process_time = process_time_total / process_count if process_count > 0 else 0

    cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(frame, f"Faces: {len(face_locations)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(frame, f"Process time: {process_time:.3f}s", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(frame, f"Detected: {len(set(detected_names))}", (frame_width - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow('Video', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p'):
        print("Video paused. Press any key to continue.")
        cv2.waitKey(-1)

cap.release()
cv2.destroyAllWindows()

print("\n===== Processing Summary =====")
print(f"Total frames processed: {process_count} of {frame_count}")
print(f"Average processing time per frame: {avg_process_time:.3f} seconds")
print(f"Total people detected: {len(attendance_record)}")
print("People detected:", ", ".join(list(attendance_record.keys())))
print("Video processing complete")