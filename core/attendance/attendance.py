import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import time

# Path for face image database
path = 'loksabha-img'
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

def findEncodings(images):
    encodeList = []
    for i, img in enumerate(images):
        print(f"Processing image {i+1}/{len(images)}")
        # Convert to RGB as face_recognition uses RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Detect faces in the image
        face_locations = face_recognition.face_locations(img_rgb)
        if not face_locations:
            print(f"No face detected in image for {classNames[i]}. Please use a clearer image.")
            continue
            
        # Use the first face found
        try:
            encoding = face_recognition.face_encodings(img_rgb, face_locations)[0]
            encodeList.append(encoding)
            print(f"Successfully encoded: {classNames[i]}")
        except IndexError:
            print(f"Could not encode face for {classNames[i]}")
            
    return encodeList, [classNames[i] for i in range(len(encodeList))]

def markAttendance(name):
    # Make sure we have a name
    if not name:
        return
        
    # Create the CSV file if it doesn't exist
    if not os.path.isfile('Attendance.csv'):
        with open('Attendance.csv', 'w') as f:
            f.write("Name,Time,Date\n")
    
    # Mark attendance
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        
        for line in myDataList:
            entry = line.strip().split(',')
            if entry and len(entry) > 0:
                nameList.append(entry[0])
                
        # Get current time
        now = datetime.now()
        dtString = now.strftime('%H:%M:%S')
        dateString = now.strftime('%Y-%m-%d')
        
        # Only add if not already present today
        if name not in nameList:
            f.writelines(f'\n{name},{dtString},{dateString}')
            print(f"Marked attendance for: {name}")

# Initialize known faces
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

# Video source - change to your video file path
video_path = 'videos/videoyt-lokshab.mp4'  # Change this to your video file path
print(f"Opening video: {video_path}")

cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print(f"Error: Could not open video {video_path}")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Video dimensions: {frame_width}x{frame_height}, FPS: {fps}")

# Variables for processing
frame_count = 0
last_recognition_time = 0
recognition_cooldown = 5  # seconds between attendance markings
skip_frames = 5  # Process every 5th frame for better performance

print("Starting video processing. Press 'q' to quit.")

while True:
    # Read a frame
    ret, frame = cap.read()
    
    # If frame reading failed, break
    if not ret:
        print("End of video or failed to read frame")
        break
    
    frame_count += 1
    
    # Skip frames to improve performance
    if frame_count % skip_frames != 0:
        # Display frame but skip processing
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue
    
    # Create a smaller image for faster processing
    # Adjust scale based on video resolution - smaller scale for higher resolution
    scale = 0.5 if frame_width <= 640 else 0.25
    small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
    
    # Convert from BGR to RGB (face_recognition uses RGB)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    
    # Find face locations in the current frame
    face_locations = face_recognition.face_locations(rgb_small_frame)
    
    # Check if any faces were found
    if face_locations:
        # Get face encodings
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
        # Process each face
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Scale back face locations to original size
            top = int(top / scale)
            right = int(right / scale)
            bottom = int(bottom / scale)
            left = int(left / scale)
            
            # Compare with known faces - using a slightly higher tolerance for video
            matches = face_recognition.compare_faces(encodeListKnown, face_encoding, tolerance=0.6)
            
            # Calculate face distances
            face_distances = face_recognition.face_distance(encodeListKnown, face_encoding)
            
            # Get best match if any
            name = "Unknown"
            confidence = 0
            
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                confidence = 1 - face_distances[best_match_index]
                
                if matches[best_match_index] and confidence > 0.4:  # Confidence threshold
                    name = valid_names[best_match_index].upper()
                    
                    # Mark attendance with cooldown
                    current_time = time.time()
                    if current_time - last_recognition_time > recognition_cooldown:
                        markAttendance(name)
                        last_recognition_time = current_time
            
            # Draw face rectangle
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            
            # Draw name label with confidence
            label = f"{name} ({confidence:.2f})" if name != "Unknown" else "Unknown"
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, label, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
    
    # Display frame count
    cv2.putText(frame, f"Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Display the frame
    cv2.imshow('Video', frame)
    
    # Break loop on 'q' press - use a longer wait for slower videos
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
print("Video processing complete")