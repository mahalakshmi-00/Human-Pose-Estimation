from flask import Flask, request, render_template, Response
import cv2
import mediapipe as mp
import numpy as np

app = Flask(__name__)

# Initialize MediaPipe Pose solution
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Initialize video capture
cap = None

def generate_frames():
    global cap
    while cap is not None:
        ret, img = cap.read()
        if not ret:
            break

        # Resize image/frame
        img = cv2.resize(img, (600, 400))

        # Do Pose detection
        results = pose.process(img)
        # Draw the detected pose on the video frame
        mp_draw.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                               mp_draw.DrawingSpec((255, 0, 0), 2, 2),
                               mp_draw.DrawingSpec((255, 0, 255), 2, 2))

        # Extract and draw pose on a plain white image
        h, w, c = img.shape
        opImg = np.zeros([h, w, c])
        opImg.fill(255)

        # Draw extracted pose on a black and white image
        mp_draw.draw_landmarks(opImg, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                               mp_draw.DrawingSpec((255, 0, 0), 2, 2),
                               mp_draw.DrawingSpec((255, 0, 255), 2, 2))

        # Display pose on the original video/live stream
        cv2.imshow("Pose Estimation", img)
        # Display extracted pose on a blank image
        cv2.imshow("Extracted Pose", opImg)

        # Encode the frame as JPEG and yield it for video streaming
        ret, buffer = cv2.imencode('.jpg', img)
        if ret:
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    global cap
    if request.method == 'POST' and 'file' in request.files:
        file = request.files['file']
        if file.filename != '':
            # Save the uploaded file temporarily
            file_path = 'temp_video.mp4'
            file.save(file_path)
            cap = cv2.VideoCapture(file_path)
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
