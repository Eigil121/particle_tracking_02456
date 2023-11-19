import lvpyio as lv
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import glob
from torchvision import transforms


class CustomDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(glob.glob(self.root + '/B*.im7')) - 4 # index by middle image, remove 2 images from start and end

    def __getitem__(self, idx):
        # Generate your data here
        # For example, let's generate a random tensor as data
        
        image_series = torch.zeros(5, 616, 2048)

        # Load images
        for i in range(0, 5):
            img = lv.read_buffer(self.root + '/B' + str(idx + i + 1).zfill(5) + '.im7')
            img = torch.from_numpy(img[0].as_masked_array().data) # Currently hardcoded to camera 1

            image_series[i, :, :] = img

        if self.transform:
            image_series = self.transform(image_series)


        return image_series, ((torch.rand_like(image_series[2,:,:]).unsqueeze(0) > 0.95)*1.0) # TODO: Add labels

upper_lower = (100, 300)

cutout_transform = transforms.Lambda(lambda x: x[:, upper_lower[0]:upper_lower[1], :]) 

# Define a transform if needed
transform = transforms.Compose([
    cutout_transform
])

def load_dataset(data_dir, batch_size):
    # Create an instance of your custom dataset
    custom_dataset = CustomDataset(data_dir, transform=transform)

    # Create a DataLoader for your custom dataset
    custom_dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)

    return custom_dataloader

