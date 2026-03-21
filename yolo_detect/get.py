import cv2
import mediapipe as mp
import os

# -----------------------------
# 初始化 MediaPipe Pose
# -----------------------------
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False)

# -----------------------------
# 打开摄像头
# -----------------------------
cap = cv2.VideoCapture(0)

# -----------------------------
# 设置动作标签
# -----------------------------
label = input("输入本次录制动作标签（stand/sit/walk/wave/reach）: ")

# -----------------------------
# 设置图片保存路径
# -----------------------------
image_dir = f"images_{label}"
os.makedirs(image_dir, exist_ok=True)
frame_id = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # BGR -> RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        # 绘制关键点（可选）
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # -----------------------------
        # 保存每帧图片，文件名带上 label 和 frame_id
        # -----------------------------
        image_path = os.path.join(image_dir, f"{label}_{frame_id:04d}.jpg")
        cv2.imwrite(image_path, frame)
        frame_id += 1

    # 可选：显示窗口
    cv2.imshow("MediaPipe Pose", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -----------------------------
# 释放资源
# -----------------------------
cap.release()
cv2.destroyAllWindows()
pose.close()