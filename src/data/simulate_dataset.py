import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader



CamDataNormalizedCumalative = torch.from_numpy(np.load('CamDataNormalizedCumalative.npy'))

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
            print(i)
            Label_domain = sim_domain.unsqueeze(0)
            threshold = 20
            binary_mask = (sim_domain >= threshold).float()
            print(binary_mask)
        #sim_domain.unsqueeze(0)
        #positions_list.append(positions)
    #print(positions_list)
    dataset.append((sim_domain_plustime,binary_mask))
    return dataset


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
