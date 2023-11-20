import torch
import matplotlib.pyplot as plt
import numpy as np
from src.data.simulate_dataset import generate_data
from src.models.architectures import SimpleCNN
from src.data.particle_dataset import load_dataset as load_dataset_real


def visualize_output(model, sample_image, sample_mask):
    model.eval()
    with torch.no_grad():
        output = model(sample_image.unsqueeze(0))
        output = output.squeeze(0)

    # Plot the model output
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

if __name__ == "__main__":
    # Load the model's state dictionary
    state_dict = torch.load('models/Simple_CNN.pth')

    model = SimpleCNN()

    # Load the state dictionary into the model
    model.load_state_dict(state_dict)
    model.eval()

    #visualize the model output for a new image
    sample_image, sample_mask = generate_data()

    #visualize the model output for an image from the real data
   # data_dir = 'C:/Users/farim/Desktop/particle_tracking_02456/data/raw/batch1'

    #load one series of images
    #data_loader = load_dataset_real(data_dir, 1)
    #test_image = next(iter(data_loader))
    
    #visualize_output(model, test_image, sample_mask)


    visualize_output(model, sample_image, sample_mask)

