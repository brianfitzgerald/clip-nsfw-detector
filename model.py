import torch.nn as nn
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
import numpy as np


class H14_NSFW_Detector(nn.Module):
    def __init__(self, input_size=1024):
        super().__init__()
        self.input_size = input_size
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)


x = torch.from_numpy(np.load("train_x.npy"))
y = torch.from_numpy(np.load("train_y.npy"))
train_dataset = TensorDataset(x, y)

# Define the data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

model = H14_NSFW_Detector()
criterion = nn.MSELoss()
optimizer = Adam(model.parameters())

num_epochs = 10

# Training loop
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        # Clear gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Compute the loss
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
