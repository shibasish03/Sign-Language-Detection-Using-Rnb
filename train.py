import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ====== CONFIG ======
BATCH_SIZE = 32
EPOCHS = 500
# LEARNING_RATE = 0.001
DATASET_PATH = "npy_data"  # Folder with .npy files organized by class
MODEL_SAVE_PATH = "rnn_data.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# ====== CUSTOM DATASET CLASS ======
class KeypointDataset(Dataset):
    def __init__(self, root_dir):
        self.data = []
        self.labels = []
        self.class_map = {}

        for idx, class_name in enumerate(sorted(os.listdir(root_dir))):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            self.class_map[idx] = class_name
            for file in os.listdir(class_dir):
                if file.endswith(".npy"):
                    self.data.append(os.path.join(class_dir, file))
                    self.labels.append(idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        keypoints = np.load(self.data[idx])
        keypoints = torch.tensor(keypoints, dtype=torch.float32)
        keypoints = keypoints.view(21, 3)  # (timesteps=21, features=3)
        label = self.labels[idx]
        return keypoints, label

# ====== RNN MODEL FOR KEYPOINTS ======
class KeypointRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(KeypointRNN, self).__init__()
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                           num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.rnn(x)  # x: (B, 21, 3)
        out = self.fc(out[:, -1, :])  # last time step output
        return out

# ====== LOAD DATASET ======
dataset = KeypointDataset(DATASET_PATH)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
NUM_CLASSES = len(set(dataset.labels))

# ====== INITIALIZE MODEL, LOSS, OPTIMIZER ======
model = KeypointRNN(input_size=3, hidden_size=128, num_layers=2, num_classes=NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), )#lr=LEARNING_RATE)

# ====== TRAINING LOOP ======
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print(f"Epoch {epoch + 1}/{EPOCHS} | Loss: {running_loss / len(train_loader):.4f} | Accuracy: {acc:.2f}%")

# ====== SAVE MODEL ======
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"✅ Trained RNN model saved to {MODEL_SAVE_PATH}")
print(f"📂 Classes: {dataset.class_map}")
