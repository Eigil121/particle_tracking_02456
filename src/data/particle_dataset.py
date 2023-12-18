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
    def __init__(self, transform=None):
        self.transform = transform

    def __len__(self):
        index_count = 0

        batches = [d for d in os.listdir("data/interim/") if not d.startswith('.')]
        # remove batch 4 if it exists as it only has empty images
        if "batch4" in batches:
            batches.remove("batch4")

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

        sample_info = {"batch": sample[0], "camera": sample[1], "image": sample[2]}
        return image_series, torch.zeros_like(image_series[2,:,:]).unsqueeze(0), sample_info



class Particle_dataset_supervised(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        
        mask_paths = glob.glob("data/interim/masks/batch*cam*_image*.png" ) # Find all masks
        pattern = re.compile(r"batch(\d+)_cam(\d+)_image(\d+).png") # Define pattern

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
        mask = torch.from_numpy(mask)
        mask = mask.unsqueeze(0)

        if self.transform:
            image_series = self.transform(image_series)
            mask = self.transform(mask)

        return image_series, mask, sample_info



class Particle_dataset_supervised_eval(Particle_dataset_supervised):

    def __init__(self, transform=None, portion=0.8, train=True):
        super().__init__(transform)
        self.portion = portion
        self.train = train

    def __getitem__(self, idx):
        # Generate your data here
        # For example, let's generate a random tensor as data

        image_series, mask, sample_info = super().__getitem__(idx)

        image_length = image_series.shape[2]

        if self.train:
            image_series = image_series[:, :, :int(image_length*self.portion)]
            mask = mask[:, :, :int(image_length*self.portion)]
        else:
            image_series = image_series[:, :, int(image_length*self.portion):]
            mask = mask[:, :, int(image_length*self.portion):]

        return image_series, mask, sample_info

class Particle_dataset_unsupervised(Particle_dataset_inference):
    def __init__(self, transform=None):
        super().__init__(transform)
        self.transform = transform
    

    def __getitem__(self, idx):
        
        image_series, _, sample_info = super().__getitem__(idx)

        # make middle image the label

        label = image_series[2, :, :].unsqueeze(0)

        # Remove middle image from image series
        image_series = torch.cat((image_series[:2, :, :], image_series[3:, :, :]), dim=0)

        return image_series, label, sample_info

def load_dataset(batch_size, dataset_type = "supervised"):

    if dataset_type == "inference":
        dataset = Particle_dataset_inference(transform=None)

    elif dataset_type == "supervised":
        dataset = Particle_dataset_supervised(transform=None)
    
    elif dataset_type == "train":
        dataset = Particle_dataset_supervised_eval(transform=None, train=True)
    
    elif dataset_type == "eval":
        dataset = Particle_dataset_supervised_eval(transform=None, train=False)
    
    elif dataset_type == "unsupervised":
        dataset = Particle_dataset_unsupervised(transform=None)
    
    else:
        Particle_dataset_inference(transform=None)

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
    data_loader = load_dataset(1, dataset_type = "unsupervised")

    for i, (images, labels, image_info) in enumerate(data_loader):
        print(images.shape)
        print(labels.shape)
        
        # Plot the first image
        plt.imshow(images[0, 2, :, :].numpy(), cmap='gray')
        plt.show()

        # Plot the first mask
        plt.imshow(labels[0, 0, :, :].numpy(), cmap='gray')
        plt.show()

        
        

