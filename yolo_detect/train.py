from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch

# -----------------------------
# 数据增强 & 预处理
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 假设你的图片按文件夹存放，每个动作一个文件夹
train_dataset = datasets.ImageFolder("train_data", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = datasets.ImageFolder("test_data", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32)

# -----------------------------
# 使用预训练模型
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=True)   # 用 ResNet18
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)  # 替换最后一层
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# -----------------------------
# 训练循环
# -----------------------------
for epoch in range(20):
    model.train()
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch} done")