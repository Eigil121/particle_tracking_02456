import lvpyio as lv
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import glob
from torchvision import transforms
import os


class Particle_dataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.transform = transform

    def __len__(self):
        index_count = 0

        batches = [d for d in os.listdir("data/interim/") if not d.startswith('.')]
        self.batch_dict = {}

        for batch in batches:
            for camera in range(4):
                for img_idx in range(0, len(glob.glob("data/interim/" + batch + "/cam" + str(camera) + "/*.png")) - 4): # index by middle image, remove 2 images from start and end
                    self.batch_dict[index_count] = (batch, str(camera), img_idx + 1)
                    index_count += 1

        return index_count
    

    def __getitem__(self, idx):
        # Generate your data here
        # For example, let's generate a random tensor as data
        
        sample = self.batch_dict[idx]

        # Load images
        for i in range(0, 5):
            # Load png as torch tensor
            img = plt.imread("data/interim/" + sample[0] + "/cam" + sample[1] + "/B" + str(sample[2] + i).zfill(5) + ".png")
            
            if i == 0:
                image_series = torch.zeros(5, img.shape[0], img.shape[1])

            img = torch.from_numpy(img)
            img = img.unsqueeze(0)

            image_series[i, :, :] = img

        if self.transform:
            image_series = self.transform(image_series)


        return image_series, ((torch.rand_like(image_series[2,:,:]).unsqueeze(0) > 0.95)*1.0) * 1.0 # TODO: Add labels

"""
upper_lower = (100, 300)

cutout_transform = transforms.Lambda(lambda x: x[:, upper_lower[0]:upper_lower[1], :]) 

# Define a transform if needed
transform = transforms.Compose([
    cutout_transform
])
"""

def load_dataset(data_dir, batch_size):
    # Create an instance of your custom dataset
    custom_dataset = Particle_dataset(data_dir, transform=None)

    # Create a DataLoader for your custom dataset
    custom_dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)

    return custom_dataloader

if __name__ == "__main__":

    import matplotlib
    matplotlib.use('TkAgg') 
    # Test the dataset
    data_dir = 'data/interim/'
    data_loader = load_dataset(data_dir, 3)

    for i, (images, labels) in enumerate(data_loader):
        print(images.shape)
        print(labels.shape)
        
        # Plot the first image
        plt.imshow(images[0, 2, :, :].numpy(), cmap='gray')
        plt.show()

        
        break

