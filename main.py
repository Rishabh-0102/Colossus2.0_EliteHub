from app import create_app
from app.camera_handler import start_video_capture

# Start the camera thread before Flask app starts
start_video_capture()

# Create the app (this registers the blueprint inside __init__.py)
app = create_app()


if __name__ == "__main__":
    app.run(debug=True)
