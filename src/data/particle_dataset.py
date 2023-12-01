import lvpyio as lv
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import glob
from torchvision import transforms
import os
import re

class Particle_dataset_inference(Dataset):
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



class Particle_dataset_supervised(Dataset):
    def __init__(self, data_dir, transform=None):
        self.transform = transform
        
        mask_paths = glob.glob("data/interim/masks/batch*cam*_image*.png" ) # Find all masks
        pattern = re.compile(r"batch(\d+)_cam(\d+)_image(\d+).png") # Define pattern

        mask_paths = mask_paths[1:] # Remove Joachim's mask

        self.image_info = []
        for mask_path in mask_paths:
            match = pattern.search(mask_path)
            if match:
                batch, camera, image = match.groups()
                self.image_info.append({"batch": batch, "camera": camera, "image": image, "mask_path": mask_path})

    def __len__(self):
        return len(self.image_info)
    

    def __getitem__(self, idx):
        # Generate your data here
        # For example, let's generate a random tensor as data

        sample_info = self.image_info[idx]

        # Load images
        for i in range(0, 5):
            # Load png as torch tensor
            img = plt.imread("data/interim/batch" + sample_info["batch"] + "/cam" + sample_info["camera"] + "/B" + str(int(sample_info["image"]) + i - 2).zfill(5) + ".png")
            
            if i == 0:
                image_series = torch.zeros(5, img.shape[0], img.shape[1])

            img = torch.from_numpy(img)
            img = img.unsqueeze(0)

            image_series[i, :, :] = img

        # Read mask as torch tensor
        mask = plt.imread(sample_info["mask_path"])
        mask = torch.from_numpy(mask)//255
        mask = mask.unsqueeze(0)

        if self.transform:
            image_series = self.transform(image_series)
            mask = self.transform(mask)


        return image_series, mask


def load_dataset(data_dir, batch_size, dataset_type = "supervised"):

    if dataset_type == "inference":
        dataset = Particle_dataset_inference(data_dir, transform=None)

    elif dataset_type == "supervised":
        dataset = Particle_dataset_supervised(data_dir, transform=None)
    
    else:
        Particle_dataset_inference(data_dir, transform=None)

    # Create a DataLoader for your custom dataset
    custom_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return custom_dataloader

if __name__ == "__main__":

    # For the Linux guy whos mpl backend is not working
    import platform
    if platform.system() == 'Linux':
        import matplotlib
        matplotlib.use('TkAgg') 

    # Test the dataset
    data_dir = 'data/interim/'
    data_loader = load_dataset(data_dir, 1, dataset_type = "supervised")

    for i, (images, labels) in enumerate(data_loader):
        print(images.shape)
        print(labels.shape)
        
        # Plot the first image
        #plt.imshow(images[0, 2, :, :].numpy(), cmap='gray')
        #plt.show()

        
        

