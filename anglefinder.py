import os
import cv2
import mediapipe as mp
import numpy as np

# Input and output folder names
body_parts = ["Triceps", "Shoulders", "Chest","Legs","Back","Biceps"]
input_base_path = "./"
output_base_path = "finalangles"

# Mediapipe setup
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Show only key joints
key_joints = [
    "left_wrist", "right_wrist", "left_elbow", "right_elbow", "left_shoulder",
    "right_shoulder", "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "spine", "head"
]

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
    den = np.linalg.norm(ab) * np.linalg.norm(cb)
    if den < 1e-6:
        return 0.0
    cosine_angle = np.dot(ab, cb) / den
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    with mp_pose.Pose(model_complexity=1) as pose:
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
                        # Optional: mark joint
                        mid_idx = idx[1]
                        cx, cy = int(lm[mid_idx].x * width), int(lm[mid_idx].y * height)
                        cv2.circle(image, (cx, cy), 5, (255, 255, 255), -1)
                    except:
                        continue
                # Draw sidebar
                start_x, start_y = 30, 40
                line_height = 35
                for i, (joint, angle) in enumerate(angle_display.items()):
                    text = f"{joint.replace('_',' ').title()}: {angle}°"
                    y = start_y + i * line_height
                    cv2.putText(image, text, (start_x + 1, y + 1),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
                    cv2.putText(image, text, (start_x, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                # Draw skeleton
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            out.write(image)
    cap.release()
    out.release()

# Main batch processing loop
for part in body_parts:
    input_folder = os.path.join(input_base_path, part)
    output_folder = os.path.join(output_base_path, part)
    os.makedirs(output_folder, exist_ok=True)
    if not os.path.exists(input_folder):
        print(f"Input folder {input_folder} does not exist.")
        continue
    video_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
    for video in video_files:
        inp = os.path.join(input_folder, video)
        outp = os.path.join(output_folder, f"angles_{video}")
        print(f"Processing {inp} -> {outp}")
        process_video(inp, outp)
