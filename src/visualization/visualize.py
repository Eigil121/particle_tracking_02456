import torch
import matplotlib.pyplot as plt
import numpy as np
from src.data.simulate_dataset import generate_data
from src.models.architectures import SimpleCNN
from src.data.particle_dataset import load_dataset as load_dataset_real


def visualize_output(input, mask, prediction, nmax=3):

    with torch.no_grad():
        input = input.clone().detach().to('cpu').numpy()
        mask = mask.clone().detach().to('cpu').numpy()
        prediction = prediction.clone().detach().to('cpu').numpy()

    N = input.shape[0]

    for i in range(N):
        plt.figure()
        plt.subplot(3, 1, 1)
        plt.imshow(input[i, 2, :, :], cmap='gray')
        plt.title("Input Image")
        plt.axis('off')
        plt.subplot(3, 1, 2)
        plt.imshow(prediction[i, 0, :, :], cmap='gray')
        plt.title("Output Probability Map")
        plt.axis('off')
        plt.subplot(3, 1, 3)
        plt.imshow(mask[i, 0, :, :], cmap='gray')
        plt.title("Input Mask")
        plt.axis('off')
        plt.show()

        if i == nmax - 1:
            break
        

if __name__ == "__main__":

    data = "real"

    # Load the model's state dictionary
    state_dict = torch.load('models/Simple_CNN.pth')

    model = SimpleCNN()

    # Load the state dictionary into the model
    model.load_state_dict(state_dict)
    model.eval()

    from src.data.simulate_dataset import load_dataset as load_dataset_sim
    from src.data.particle_dataset import load_dataset as load_dataset_real
    
    if data == "simulate":
        data_loader = load_dataset_sim(1)
        sample_image, sample_mask = next(iter(data_loader))
  

    elif data == "real":
        data_loader = load_dataset_real(1)
        sample_image, sample_mask = next(iter(data_loader))

    output = model(sample_image)


    visualize_output(sample_image, sample_mask, output)

