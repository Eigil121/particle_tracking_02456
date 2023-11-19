import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from src.data.particle_dataset import load_dataset
#from src.data.simulate_dataset import load_dataset
from architectures import SimpleCNN


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

data_dir = '/home/eigil/DTU/particle_tracking_02456/data/Deep Learning Turbulence Project/ML_project_data/ImgPreproc(no_subtract)/'
data_loader = load_dataset(data_dir, 3)
model = SimpleCNN()
train(model, data_loader, num_epochs=1)

#visualize the model output for a new image
sample_image, sample_mask = generate_data()
model.eval()

with torch.no_grad():
    output = model(sample_image.unsqueeze(0))
    output = output.squeeze(0)

#plot the model output
print(sample_image.shape)
plt.figure()
plt.subplot(1, 3, 1)
plt.imshow(sample_image[2].numpy(), cmap='gray')
plt.title("Input Image")
plt.axis('off')
plt.subplot(1, 3, 2)
plt.imshow(output[0].numpy(), cmap='gray')
plt.title("Output Probability Map")
plt.axis('off')
plt.subplot(1, 3, 3)
plt.imshow(sample_mask[0].numpy(), cmap='gray')
plt.title("Input Mask")
plt.axis('off')
plt.show()
