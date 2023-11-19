import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
#from src.data.particle_dataset import load_dataset
from torch.utils.data import Dataset, DataLoader



#Remove from her
CamDataNormalizedCumalative = torch.from_numpy(np.load('C:\\Users\\farim\\Desktop\\particle_tracking_02456\\src\\data\\CamDataNormalizedCumalative.npy'))

def rdNoise(CamDataNormalizedCumalative, Shape):
    if isinstance(Shape, int):
        rdNumb = torch.searchsorted(CamDataNormalizedCumalative[:, 0], torch.rand(size=(Shape,)))
        return rdNumb
    else:
        Sizelength = Shape[0] * Shape[1]
        rdNumb = torch.searchsorted(CamDataNormalizedCumalative[:, 0], torch.rand(size=(Sizelength,)))
        rdNumb = rdNumb.reshape(Shape)
        return rdNumb

def gaussian(pos, mu, params):
    x, y = pos
    x0, y0 = mu
    A, sigma = params
    return A * torch.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

def generate_images(num_images=5, x_dim=100, y_dim=100, num_gaussians=100, gauss_params=(50, 1)):
    dataset = []
    images = []
    positions_list = []
    gauss_masks = []
    #flowrate = 5+max(min(random.gauss(0, 1), 2), -2)
    #Angle = random.uniform(0, 2 * np.pi)
    #flowX = flowrate * np.cos(Angle)
    #flowY = flowrate * np.sin(Angle)
    #Init_positions = torch.zeros((2, num_gaussians))
    #Init_positions[0, :] = torch.rand(num_gaussians) * x_dim
    #Init_positions[1, :] = torch.rand(num_gaussians) * y_dim
    sim_domain_plustime = torch.zeros((num_images ,x_dim, y_dim ))
    Label_domain = torch.zeros((x_dim, y_dim ))
    for i in range(num_images):
        sim_domain = torch.zeros((x_dim, y_dim))
        positions = torch.zeros((2, num_gaussians))
        positions[0, :] = torch.randint(0, x_dim, (num_gaussians,))
        positions[1, :] = torch.randint(0, y_dim, (num_gaussians,))
        #print(positions)
        x = torch.arange(x_dim)
        y = torch.arange(y_dim)
        x, y = torch.meshgrid(x, y)

        for n in range(num_gaussians):
            sim_domain += gaussian((x, y), positions[:, n], gauss_params).t()

        # Store the images and masks as tensors with 1 channel
        #noise = torch.normal(mean=0., std=noise_stddev, size=sim_domain.shape)
        noise = rdNoise(CamDataNormalizedCumalative, sim_domain.shape)
        



        # Store the images and masks as tensors with 1 channel
        sim_domain_plustime[i,:,:] = sim_domain.unsqueeze(0) + 1.0*noise
        if i == round(num_images/2):
            #print(i)
            Label_domain = sim_domain.unsqueeze(0)
            threshold = 20
            binary_mask = (sim_domain >= threshold).float()
            #print(binary_mask)
        #sim_domain.unsqueeze(0)
        #positions_list.append(positions)
    #print(positions_list)
    dataset.append((sim_domain_plustime,binary_mask))
    #return dataset
    return sim_domain_plustime, binary_mask.unsqueeze(0)

num_images = 5
generate_data = lambda: generate_images(num_images=num_images, x_dim=100, y_dim=100, num_gaussians=100, gauss_params=(50, 1))

class CustomDataset(Dataset):
    def __init__(self, root, transform=None):
        pass

    def __len__(self):
        return 100 # index by middle image, remove 2 images from start and end

    def __getitem__(self, idx):
        # Generate your data here
        # For example, let's generate a random tensor as data
        
        image_series, mask = generate_data()

        
        return image_series, mask # TODO: Add labels



def load_dataset(data_dir, batch_size):
    # Create an instance of your custom dataset
    custom_dataset = CustomDataset(None, transform=None)

    # Create a DataLoader for your custom dataset
    custom_dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)

    return custom_dataloader

#Remove to here





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
data_loader = load_dataset(data_dir, 2)
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
