import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import time

# === CONFIG ===
IMAGE_DIR = 'imageAttendance'
ATTENDANCE_FILE = 'Attendance.csv'
VIDEO_SOURCE = './videos/vid6.mp4'  # 0 for webcam, or 'videos/video1.mp4' for file
SKIP_FRAMES = 2
COOLDOWN_SECONDS = 30
CONFIDENCE_THRESHOLD = 0.4
FACE_DETECTION_MODEL = "hog"  # or "cnn" for more accuracy

# === Setup folders ===
os.makedirs(IMAGE_DIR, exist_ok=True)


# === Load known faces ===
def load_known_faces(image_dir):
    images = []
    names = []

    print(f"\n🔍 Loading images from {image_dir}...")
    for file in os.listdir(image_dir):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(image_dir, file)
            img = cv2.imread(img_path)
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                faces = face_recognition.face_locations(img_rgb)
                if faces:
                    enc = face_recognition.face_encodings(img_rgb, faces)[0]
                    images.append(enc)
                    names.append(os.path.splitext(file)[0])
                    print(f"✅ Encoded: {file}")
                else:
                    print(f"⚠️ No face detected in {file}")
    return images, names


# === Attendance ===
def mark_attendance(name):
    if not os.path.isfile(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, 'w') as f:
            f.write("Name,Time,Date\n")

    now = datetime.now()
    time_str = now.strftime('%H:%M:%S')
    date_str = now.strftime('%Y-%m-%d')

    with open(ATTENDANCE_FILE, 'r+') as f:
        lines = f.readlines()
        names_today = [line.split(',')[0] for line in lines if date_str in line]

        if name not in names_today:
            f.writelines(f"{name},{time_str},{date_str}\n")
            print(f"📝 Attendance marked for {name}")


# === Progress Bar ===
def show_progress(current, total, bar_len=40):
    frac = current / total
    filled = int(frac * bar_len)
    return f"[{'█' * filled}{'░' * (bar_len - filled)}] {int(frac * 100)}%"


# === Video Processing ===
def process_video(known_encodings, known_names, video_source=0):
    print(f"\n🎥 Starting video: {video_source}")
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print(f"❌ Failed to open video source: {video_source}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    scale = 0.25 if width > 640 else 0.5

    print(f"Video Resolution: {width}x{height} | Processing every {SKIP_FRAMES} frames")

    attendance_tracker = {}
    frame_count = 0
    processed_count = 0
    total_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("🎞️ End of stream or failed to read.")
            break

        frame_count += 1
        if frame_count % SKIP_FRAMES != 0:
            cv2.imshow("Video", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        start_time = time.time()

        # Resize and convert
        small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Detect and encode
        locations = face_recognition.face_locations(rgb_frame, model=FACE_DETECTION_MODEL)
        encodings = face_recognition.face_encodings(rgb_frame, locations)

        names_in_frame = []

        for encoding, loc in zip(encodings, locations):
            matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=0.6)
            distances = face_recognition.face_distance(known_encodings, encoding)

            name = "Unknown"
            confidence = 0

            if len(distances) > 0:
                best_match = np.argmin(distances)
                confidence = 1 - distances[best_match]
                if matches[best_match] and confidence > CONFIDENCE_THRESHOLD:
                    name = known_names[best_match].upper()

            names_in_frame.append(f"{name} ({confidence:.2f})" if name != "Unknown" else "Unknown")

            # Mark attendance
            if name != "Unknown":
                now = time.time()
                if name not in attendance_tracker or now - attendance_tracker[name] > COOLDOWN_SECONDS:
                    mark_attendance(name)
                    attendance_tracker[name] = now

        # Draw rectangles for all detected faces
        for loc, label in zip(locations, names_in_frame):
            top, right, bottom, left = [int(p / scale) for p in loc]
            color = (0, 255, 0) if "Unknown" not in label else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, label, (left, bottom + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Show stats
        duration = time.time() - start_time
        total_time += duration
        processed_count += 1
        avg_time = total_time / processed_count

        cv2.putText(frame, f"Frame: {frame_count} | Avg time: {avg_time:.2f}s",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 255), 2)
        cv2.imshow("Video", frame)

        if total_frames > 0 and frame_count % 30 == 0:
            print(show_progress(frame_count, total_frames))

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            print("⏸️ Paused. Press any key to resume.")
            cv2.waitKey(-1)

    cap.release()
    cv2.destroyAllWindows()

    print("\n✅ Processing Done")
    print(f"Frames processed: {processed_count}")
    print(f"Average frame time: {avg_time:.3f} sec")
    print(f"People detected: {list(attendance_tracker.keys())}")


# === MAIN EXECUTION ===
if __name__ == "__main__":
    encodings, names = load_known_faces(IMAGE_DIR)
    if not encodings:
        print("❌ No valid faces found. Exiting.")
    else:
        process_video(encodings, names, VIDEO_SOURCE)
