from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch

# -----------------------------
# Data Augmentation & Preprocessing
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Assuming images are stored in folders, with one folder per action
train_dataset = datasets.ImageFolder("train_data", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = datasets.ImageFolder("test_data", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32)

# -----------------------------
# Using Pretrained Model
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=True)   # Using ResNet18
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)  # Replace the last layer
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# -----------------------------
# Training Loop
# -----------------------------
for epoch in range(20):
    model.train()
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs