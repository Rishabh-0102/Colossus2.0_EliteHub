# app/camera_handler.py

import cv2
import threading
import subprocess
import os
import time
from flask import jsonify

# Global flags
analyzing = False
attendance_process = None
capture_active = False
capture_error = None
latest_frame_path = "shared_frame.jpg"
frame_ready = False

def video_capture():
    global capture_active, capture_error, frame_ready

    try:
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            capture_error = "Could not open camera. Please check connections and permissions."
            print(f"Error: {capture_error}")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        os.makedirs("output", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output/recorded_output.avi', fourcc, 20.0, (width, height))

        capture_active = True
        consecutive_errors = 0

        while capture_active:
            ret, frame = cap.read()
            if not ret:
                consecutive_errors += 1
                if consecutive_errors > 5:
                    capture_error = "Lost connection to camera and couldn't reconnect."
                    break
                cap.release()
                time.sleep(1)
                cap = cv2.VideoCapture(0)
                continue

            consecutive_errors = 0
            out.write(frame)
            cv2.imwrite(latest_frame_path, frame)
            frame_ready = True
            time.sleep(0.01)

    except Exception as e:
        capture_error = f"Error in video capture: {str(e)}"
        print(capture_error)

    finally:
        capture_active = False
        frame_ready = False
        if 'cap' in locals() and cap: cap.release()
        if 'out' in locals() and out: out.release()
        cv2.destroyAllWindows()

def start_video_capture():
    thread = threading.Thread(target=video_capture, daemon=True)
    thread.start()
    time.sleep(2)  # Let camera warm up

def get_camera_status():
    return {
        'active': capture_active,
        'error': capture_error,
        'frame_ready': frame_ready
    }

def start_analysis():
    global analyzing, attendance_process
    if analyzing:
        return "Analysis already running."
    try:
        attendance_process = subprocess.Popen([
            'python', 'analyzer/attendance.py',
            '--frame-source', 'file',
            '--frame-path', latest_frame_path
        ])
        analyzing = True
        print("Started analyzing. Process ID:", attendance_process.pid)
        return "Analysis started."
    except Exception as e:
        return f"Error starting analysis: {str(e)}"

def stop_analysis():
    global analyzing, attendance_process
    if not analyzing:
        return "Analysis not running."
    try:
        if attendance_process:
            attendance_process.terminate()
            attendance_process = None
        analyzing = False
        print("Stopped analyzing.")
        return "Analysis stopped."
    except Exception as e:
        return f"Error stopping analysis: {str(e)}"
