import cv2
import numpy as np
import time

# Load the pre-trained models for face detection
print("Loading face detection models...")
try:
    # Load the DNN model
    modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
    configFile = "deploy.prototxt"
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    use_dnn = True
    print("Using DNN face detector")
except Exception as e:
    print(f"Warning: Could not load DNN model: {e}")
    print("Falling back to Haar Cascade detector")
    # Fallback to Haar Cascade detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    use_dnn = False

# Load an additional profile face detector to catch side views
profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

# Video path
video_path = 'videos/videoyt-lokshab.mp4'
video_capture = cv2.VideoCapture(video_path)

if not video_capture.isOpened():
    print(f"Error: Could not open video at {video_path}")
    exit()

# Get video properties
fps = video_capture.get(cv2.CAP_PROP_FPS)
width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

# Set up parameters
frame_skip = 1  # Process every Nth frame (1 = every frame)
detection_interval = 3  # Perform full detection every N frames

# Performance metrics
start_time = time.time()
frames_processed = 0
faces_detected = 0

# Create window that can be resized
cv2.namedWindow('Enhanced Face Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Enhanced Face Detection', width, height)

def detect_faces(frame):
    """Detect faces in the frame using the selected method"""
    faces = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
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
                    faces.append((x1, y1, x2-x1, y2-y1, confidence))
    else:
        # Haar Cascade detection
        haar_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in haar_faces:
            faces.append((x, y, w, h, 0.7))  # Assign default confidence
            
    # Try to detect profile faces if few faces were found
    if len(faces) < 2:
        # Try both original and flipped image to catch profiles facing both directions
        profile_faces = profile_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        flipped = cv2.flip(gray, 1)
        flipped_profile_faces = profile_cascade.detectMultiScale(flipped, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # Add profile faces
        for (x, y, w, h) in profile_faces:
            faces.append((x, y, w, h, 0.6))  # Lower confidence for profile faces
            
        # Convert flipped coordinates back to original image
        frame_width = gray.shape[1]
        for (x, y, w, h) in flipped_profile_faces:
            # Flip x-coordinate: new_x = width - (x + w)
            new_x = frame_width - (x + w)
            faces.append((new_x, y, w, h, 0.6))
    
    return faces

print(f"Starting video processing: {width}x{height} at {fps} FPS")

try:
    frame_counter = 0
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
            
        # Only process every Nth frame for efficiency
        if frame_counter % frame_skip == 0:
            # Detect faces in the current frame
            detected_faces = detect_faces(frame)
            
            # Update statistics
            faces_detected += len(detected_faces)
            
            # Visualize the results
            for (x, y, w, h, conf) in detected_faces:
                # Calculate a color based on confidence (green -> red)
                color = (0, int(255 * conf), int(255 * (1 - conf)))
                
                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                
                # Display confidence score
                label = f"{conf:.2f}"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Display performance info
            elapsed_time = time.time() - start_time
            fps_text = f"FPS: {frames_processed / max(1, elapsed_time):.1f}"
            cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, (0, 0, 255), 2)
            
            # Show progress
            progress = f"Frame: {frame_counter}/{total_frames} ({100*frame_counter/max(1,total_frames):.1f}%)"
            cv2.putText(frame, progress, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, (0, 0, 255), 2)
            
            # Display the frame
            cv2.imshow('Enhanced Face Detection', frame)
            frames_processed += 1
        
        frame_counter += 1
        
        # Calculate delay to maintain original video speed
        delay = max(1, int(1000 / fps))
        key = cv2.waitKey(delay)
        
        # Handle user input
        if key == 27:  # Esc key
            break
        elif key == ord('s'):  # Skip ahead
            frame_counter += int(fps * 5)  # Skip 5 seconds
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_counter)

except Exception as e:
    print(f"Error during processing: {e}")
finally:
    # Clean up
    video_capture.release()
    cv2.destroyAllWindows()
    
    # Print summary statistics
    print("\n--- Performance Summary ---")
    print(f"Total frames processed: {frames_processed}")
    print(f"Total faces detected: {faces_detected}")
    print(f"Average faces per frame: {faces_detected / max(1, frames_processed):.2f}")
    print(f"Processing time: {time.time() - start_time:.2f} seconds")
    print(f"Average FPS: {frames_processed / max(1, time.time() - start_time):.2f}")