import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import face_recognition_models

# Function to adjust brightness and contrast
def adjust_brightness_contrast(image, brightness=0, contrast=0):
    """
    Adjusts brightness and contrast of an image
    :param image: Input image
    :param brightness: Brightness adjustment (-100 to 100)
    :param contrast: Contrast adjustment (-100 to 100)
    :return: Adjusted image
    """
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow
        
        # Apply brightness adjustment
        image = cv2.addWeighted(image, alpha_b, image, 0, gamma_b)
    
    if contrast != 0:
        alpha_c = float(131 * (contrast + 127)) / (127 * (131 - contrast))
        gamma_c = 127 * (1 - alpha_c)
        
        # Apply contrast adjustment
        image = cv2.addWeighted(image, alpha_c, image, 0, gamma_c)
    
    return image

path = 'imageAttendance'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    currImg = cv2.imread(f'{path}/{cl}')
    images.append(currImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')

encodeListKnown = findEncodings(images)
print('Encoding Complete')

# Path to your MOV file
video_path = 'IMG_3016.MOV'  # Change this to your MOV file path
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}")
    exit()

# Parameters for brightness and contrast adjustment
brightness_value = 30  # Adjust as needed (range: -100 to 100)
contrast_value = 10    # Adjust as needed (range: -100 to 100)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_count/fps
print(f"Video FPS: {fps}")
print(f"Total frames: {frame_count}")
print(f"Duration: {duration:.2f} seconds")

while True:
    success, img = cap.read()
    
    # If the video ends, break the loop
    if not success:
        print("End of video file")
        break
    
    # Apply brightness and contrast adjustment
    img = adjust_brightness_contrast(img, brightness_value, contrast_value)
    
    # Resize for faster processing
    imgs = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)

    facesCurrFrame = face_recognition.face_locations(imgs)
    encodesCurrFrame = face_recognition.face_encodings(imgs, facesCurrFrame)

    for encodeFace, faceLoc in zip(encodesCurrFrame, facesCurrFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        # print(faceDis)  # Commented out to reduce console spam
        
        if len(faceDis) > 0:  # Check if there are any face distances
            matchIndex = np.argmin(faceDis)
            
            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                print(f"Recognized: {name}")
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                markAttendance(name)

    # Display the video with face boxes
    cv2.imshow('Video', img)
    
    # Control playback speed - wait time in milliseconds
    # Adjust the wait time to control video playback speed
    # Lower values = faster playback, higher values = slower playback
    wait_time = max(1, int(1000/fps))  # Default to real-time playback
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(wait_time) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()