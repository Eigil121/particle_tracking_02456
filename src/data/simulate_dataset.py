import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader



def generate_images(num_images=5, x_dim=100, y_dim=100, num_gaussians=100, gauss_params=(50, 1)):
    pass


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
