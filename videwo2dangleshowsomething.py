import cv2
import mediapipe as mp
import numpy as np

# Input and output video paths
input_path = "dumbellpress.mp4"   # Replace with your video filename
output_path = "2ddumbellpress.mp4"

# Mediapipe setup
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Show only key joints
key_joints = ["left_wrist", "right_wrist", "left_elbow", "right_elbow", "left_shoulder", "right_shoulder"]

# Joints angle mapping
angle_joints = {
    "left_elbow": ["LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"],
    "right_elbow": ["RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"],
    "left_shoulder": ["LEFT_ELBOW", "LEFT_SHOULDER", "LEFT_HIP"],
    "right_shoulder": ["RIGHT_ELBOW", "RIGHT_SHOULDER", "RIGHT_HIP"],
    "left_hip": ["LEFT_SHOULDER", "LEFT_HIP", "LEFT_KNEE"],
    "right_hip": ["RIGHT_SHOULDER", "RIGHT_HIP", "RIGHT_KNEE"],
    "left_knee": ["LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"],
    "right_knee": ["RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"],
    "left_wrist": ["LEFT_ELBOW", "LEFT_WRIST", "LEFT_INDEX"],
    "right_wrist": ["RIGHT_ELBOW", "RIGHT_WRIST", "RIGHT_INDEX"],
    "left_ankle": ["LEFT_KNEE", "LEFT_ANKLE", "LEFT_HEEL"],
    "right_ankle": ["RIGHT_KNEE", "RIGHT_ANKLE", "RIGHT_HEEL"],
    "spine": ["LEFT_HIP", "LEFT_SHOULDER", "RIGHT_HIP"],
    "head": ["LEFT_SHOULDER", "NOSE", "RIGHT_SHOULDER"]
}

# 2D angle calculation
def calculate_angle(a, b, c):
    a = np.array([a[0], a[1]])
    b = np.array([b[0], b[1]])
    c = np.array([c[0], c[1]])
    ab = a - b
    cb = c - b
    cosine_angle = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

# Video capture
cap = cv2.VideoCapture(input_path)

# Get video properties
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Output video writer
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    angle_display = {}

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark

        for joint, triplet in angle_joints.items():
            if joint not in key_joints:
                continue

            try:
                idx = [getattr(mp_pose.PoseLandmark, j).value for j in triplet]
                coords = [(lm[i].x * width, lm[i].y * height) for i in idx]
                angle = calculate_angle(*coords)
                angle_display[joint] = int(angle)

                # Optional circle marker (still drawn on body)
                mid_idx = idx[1]
                cx, cy = int(lm[mid_idx].x * width), int(lm[mid_idx].y * height)
                cv2.circle(image, (cx, cy), 5, (255, 255, 255), -1)

            except:
                continue

        # Draw sidebar text
        start_x, start_y = 30, 40
        line_height = 35
        for i, (joint, angle) in enumerate(angle_display.items()):
            text = f"{joint.replace('_',' ').title()}: {angle}°"
            y = start_y + i * line_height
            # Black border for readability
            cv2.putText(image, text, (start_x + 1, y + 1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
            # White text
            cv2.putText(image, text, (start_x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Draw skeleton
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    out.write(image)
    cv2.imshow("Angle Tracker", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
