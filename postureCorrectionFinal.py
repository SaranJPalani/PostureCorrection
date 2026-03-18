import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time

# ---------------------- User Input ----------------------
exercise_name = input("Enter Exercise Name (case-sensitive as in CSV): ").strip()

# ---------------------- Load Exercise Config ----------------------
df = pd.read_csv("exercise_metrics.csv", delimiter=",")

if exercise_name not in df["Exercise"].values:
    print(f"Exercise '{exercise_name}' not found in CSV.")
    exit()

exercise_row = df[df["Exercise"] == exercise_name].iloc[0]
required_metrics = exercise_row["Required_Metrics"].split(",")

metric_bounds = {}
for metric in required_metrics:
    val = str(exercise_row.get(metric, ""))
    if pd.isna(val) or val.strip() == "":
        continue
    try:
        range_part, buffer = val.split(",")
        start, end = map(float, range_part.split("-"))
        metric_bounds[metric] = {"start": start, "end": end, "buffer": float(buffer)}
    except:
        print(f"Invalid format for {metric}: {val}")

# ---------------------- Mediapipe Setup ----------------------
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

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

# ---------------------- 2D Angle Calculation ----------------------
def calculate_angle(a, b, c):
    a, b, c = np.array(a[:2]), np.array(b[:2]), np.array(c[:2])  # Use only x and y
    ab = a - b
    cb = c - b
    cosine_angle = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

# ---------------------- State Variables ----------------------
rep_count = 0
stage = "start"
feedback = ""
left_start_time = None
GRACE_PERIOD = 2.5  # seconds
start_threshold = 0.6  # 60% joints within buffer considered valid

# ---------------------- Webcam Feed ----------------------
cap = cv2.VideoCapture(2)  # Change to 1 or 2 if needed

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    h, w, _ = image.shape
    angles = {metric: 0 for metric in required_metrics}

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark

        for metric in required_metrics:
            if metric not in angle_joints or metric not in metric_bounds:
                continue
            try:
                j1, j2, j3 = angle_joints[metric]
                idx = [getattr(mp_pose.PoseLandmark, j).value for j in [j1, j2, j3]]
                coords = [(lm[i].x * w, lm[i].y * h) for i in idx]  # 2D only
                angle = calculate_angle(*coords)
                angles[metric] = angle
            except:
                continue

        within_start_count = sum(
            abs(angles[m] - metric_bounds[m]["start"]) <= metric_bounds[m]["buffer"]
            for m in required_metrics if m in metric_bounds
        )
        within_end_count = sum(
            abs(angles[m] - metric_bounds[m]["end"]) <= metric_bounds[m]["buffer"]
            for m in required_metrics if m in metric_bounds
        )

        all_in_start = within_start_count / len(required_metrics) >= start_threshold
        all_in_end = within_end_count / len(required_metrics) >= start_threshold

        current_time = time.time()

        if stage == "start":
            if all_in_end:
                stage = "end"
                feedback = ""
                left_start_time = None
            elif not all_in_start:
                if left_start_time is None:
                    left_start_time = current_time
                elif current_time - left_start_time > GRACE_PERIOD:
                    feedback = "You left start but didn't reach end. Try again."
            else:
                left_start_time = None
                feedback = ""

        elif stage == "end":
            if all_in_start:
                stage = "start"
                rep_count += 1
                feedback = "Good rep!"
                left_start_time = None

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Display Rep Count
    cv2.putText(image, f"Reps: {rep_count}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

    # Display Feedback
    if feedback:
        cv2.putText(image, feedback, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow(f"{exercise_name} Tracker (2D)", image)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
