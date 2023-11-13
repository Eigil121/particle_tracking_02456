import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from src.data.particle_dataset import load_dataset



class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(5, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 1, kernel_size=3, padding=1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.activation(self.conv3(x))
        return x



def train(model, dataloader, num_epochs=1, learning_rate=0.001):
    # Move model to device (CPU/GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define loss function and optimizer
    #criterion = nn.BCEWithLogitsLoss()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    train_loss_history = []

    for epoch in range(num_epochs):
        model.train()

        for i, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)

            # Move tensors to device (CPU/GPU)
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            loss.backward()
            optimizer.step()

            # Print training information
            print(f"Epoch {epoch + 1}/{num_epochs}, Batch {i + 1}/{len(dataloader)}, Loss: {loss.item():.4f}")

            # Append current loss to history
            train_loss_history.append(loss.item())

data_dir = 'C:/Users/farim/Desktop/particle_tracking_02456/data/'
data_loader = load_dataset(data_dir, 3)
model = SimpleCNN()
train(model, data_loader, num_epochs=1)

