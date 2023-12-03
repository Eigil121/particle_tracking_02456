import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import random
from PIL import Image



#load the data with os
#CamDataNormalizedCumalative = torch.from_numpy(np.load(os.path.join(os.getcwd(), 'data', 'CamDataNormalizedCumalative.npy')))
#CamDataNormalizedCumalative = torch.from_numpy(np.load('CamDataNormalizedCumalative.npy'))
CamDataNormalizedCumalative = torch.from_numpy(np.load(os.path.join(os.getcwd(), 'src/data', 'CamDataNormalizedCumalative.npy')))
ParticleZoo = torch.from_numpy(np.load(os.path.join(os.getcwd(), 'src/data', 'ParticleZoo.npy')))
#ParticleZoo = torch.from_numpy(np.load('src/data/ParticleZoo.npy'))
ParticleBoxSize = 14
NParticlesZoo = int(np.floor((ParticleZoo.shape[1])/ParticleBoxSize))

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

def AddParticle(SimDomain,pos,ZooId):
    x, y = pos
    Nx=len(SimDomain[:,0])
    Ny=len(SimDomain[0,:])
    #print(Nx,Ny)
    img2_path = 'data/interim/batch1/cam0/B00002.png'
    img2 = Image.open(img2_path)
    img2 = np.array(img2)
    # ParticleZoo = np.zeros((ParticleBoxSize,ParticleBoxSize*NParticlesZoo))
    # #q = int(random.randint(0, NParticles))
    # #print(sortedx[q])
    # #print(NParticlesZoo)
    # for q in range(NParticlesZoo):
    #     imgx = img2[int(sortedy[q]):int(sortedy[q])+ParticleBoxSize,int(sortedx[q]):int(sortedx[q])+ParticleBoxSize]
    #     maskx = mask[int(sortedy[q]):int(sortedy[q])+ParticleBoxSize,int(sortedx[q]):int(sortedx[q])+ParticleBoxSize]
    #     ParticleZoo[:,ParticleBoxSize*q:ParticleBoxSize*(q+1)] = (imgx*maskx)[:,:]
    # Plot the image in big format
    #ParticleZoo[:,14*ZooId:14*(ZooId+1)]
    #print(SimDomain.shape,(x,x+14,y,y+14))
    #print('hui')
    # if x+14<Nx and y+14<Ny:
        #print('hi')
        #x=Nx-14
    #print( SimDomain[x:x+14,y:y+14].shape,torch.tensor(ParticleZoo[:,14*ZooId:14*(ZooId+1)]).shape)
    if torch.tensor(ParticleZoo[:,14*ZooId:14*(ZooId+1)]).shape == SimDomain[x:x+14,y:y+14].shape:
        SimDomain[x:x+14,y:y+14] += torch.tensor(ParticleZoo[:,14*ZooId:14*(ZooId+1)])
        
    #SimDomain[x:x+14,y:y+14] += ParticleZoo[:,14*ZooId:14*(ZooId+1)]
    #print(SimDomain.shape)
    return SimDomain

def generate_images(num_images=5, x_dim=100, y_dim=100, NParticles=100, gauss_params=(60, 1), meanflow=15):
    dataset = []
    images = []
    positions_list = []
    gauss_masks = []
    flowrate = meanflow + max(min(np.random.normal(0, 1), 2), -2)
    Angle = np.random.uniform(0, 2 * np.pi)
    flowX = flowrate * np.cos(Angle)
    flowY = flowrate * np.sin(Angle)
    positions = torch.zeros((5, 2, NParticles))
    positions[0, 0, :] = torch.rand(NParticles) * (x_dim+meanflow*(num_images+1)+2*14)#-meanflow*(num_images+1)/2-14
    positions[0, 1, :] = torch.rand(NParticles) * (y_dim+meanflow*(num_images+1)+2*14)#-meanflow*(num_images+1)/2-14
    domainextention = meanflow*(num_images+1)+2*14+20
    #print(domainextention)
    sim_domain_plustime = torch.zeros((num_images ,x_dim, y_dim ))
    sim_domain_plustime_withNoise = torch.zeros((num_images ,x_dim, y_dim ))
    sim_domain_plustime_extended = torch.zeros((num_images ,x_dim+domainextention, y_dim+domainextention ))
    #print(positions)
    Label_domain = torch.zeros((x_dim, y_dim ))
    random_numbers = [random.randint(0, NParticlesZoo) for _ in range(NParticles)]
    for i in range(num_images):
        sim_domain = torch.zeros((x_dim, y_dim))
        #positions = torch.zeros((2, NParticles))
        positions[i, 0, :] = positions[0, 0, :]+flowX*i
        positions[i, 1, :] = positions[0, 1, :]+flowY*i
        #print(positions)
        x = torch.arange(x_dim)
        y = torch.arange(y_dim)
        x, y = torch.meshgrid(x, y)

        #print((int(positions[0, 0, 0]),int(positions[0, 1, 0])))
        for n in range(NParticles):
            sim_domain_plustime_extended[i,:,:] = AddParticle(sim_domain_plustime_extended[i,:,:],(int(positions[i, 0, n]),int(positions[i, 1, n])),ZooId=random_numbers[n])
        # plt.figure()
        # plt.imshow(sim_domain_plustime_extended[i,:,:])
        # plt.show()
        # #print(sim_domain_plustime_extended[i,int(domainextention/2):int(domainextention/2+100),int(domainextention/2):int(domainextention/2+100)].shape)
        sim_domain_plustime[i,:,:] = sim_domain_plustime_extended[i,int(domainextention/2):int(domainextention/2+100),int(domainextention/2):int(domainextention/2+100)]
        # for n in range(NParticles):
        #     gauss_paramsrnd = (gauss_params[0] + max(np.random.normal(0, 30), -60), gauss_params[1]+ max(np.random.normal(0, 0.5), -1))
        #     sim_domain += gaussian((x, y), positions[i, :, n], gauss_params).t()

        # Store the images and masks as tensors with 1 channel
        #noise = torch.normal(mean=0., std=noise_stddev, size=sim_domain.shape)
        noise = rdNoise(CamDataNormalizedCumalative, sim_domain.shape)
        



        # Store the images and masks as tensors with 1 channel
        sim_domain_plustime_withNoise[i,:,:] = sim_domain_plustime[i,:,:].unsqueeze(0) + 1.0*noise
        if i == round(num_images/2):
            #print(i,sim_domain_plustime[i,:,:].shape)
            Label_domain = sim_domain_plustime[i,:,:].unsqueeze(0)
            threshold = 20
            binary_mask = (Label_domain >= threshold).float()
            
        #sim_domain.unsqueeze(0)
        #positions_list.append(positions)
    #print(positions_list)
    binary_mask = binary_mask.unsqueeze(0)
    return sim_domain_plustime_withNoise, binary_mask[0]


num_images = 5
generate_data = lambda: generate_images(num_images=num_images, x_dim=100, y_dim=100, NParticles=100, gauss_params=(50, 1), meanflow=15)

class CustomDataset(Dataset):
    def __init__(self, root, transform=None):
        pass

    def __len__(self):
        return 10 # index by middle image, remove 2 images from start and end

    def __getitem__(self, idx):
        # Generate your data here
        # For example, let's generate a random tensor as data
        
        image_series, mask = generate_data()

        return image_series, mask, {"batch": 0, "camera": "x", "image": "x"}



def load_dataset(batch_size, dataset_type = None):

    # dataset_type is a placeholder for now. It is used for different model types, i.e. supervised, unsupervised, etc.

    # Create an instance of your custom dataset
    custom_dataset = CustomDataset(None, transform=None)

    # Create a DataLoader for your custom dataset
    custom_dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)

    return custom_dataloader
