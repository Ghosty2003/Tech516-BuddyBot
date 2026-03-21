import numpy as np
import csv
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# 参数
# -----------------------------
csv_file = "pose_data.csv"  # 你的 CSV 文件

# -----------------------------
# 读取 CSV
# -----------------------------
data = []
labels = []

with open(csv_file, newline='') as f:
    reader = csv.reader(f)
    for row in reader:
        *coords, label = row
        coords = np.array(coords, dtype=np.float32)
        data.append(coords)
        labels.append(label)

data = np.stack(data)  # shape: (num_frames, num_features)
labels = np.array(labels)

print("Data shape:", data.shape)
print("Labels shape:", labels.shape)

# -----------------------------
# 标签编码
# -----------------------------
le = LabelEncoder()
y_encoded = le.fit_transform(labels)
print("Encoded labels:", y_encoded[:10])
print("Label classes:", le.classes_)

# -----------------------------
# 保存为 .npz
# -----------------------------
np.savez("motion_dataset.npz", X=data, y=y_encoded)
print("Saved to motion_dataset.npz")