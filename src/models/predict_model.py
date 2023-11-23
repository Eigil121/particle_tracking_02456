#import click
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from src.data.particle_dataset import load_dataset as load_dataset_real
from src.models.architectures import SimpleCNN
#@click.command()
#@click.argument('model_name', type=str)
def predict_model(model_name, dataloader):

    print(f"Predicting with {model_name}...")
    model_path = 'models/' + model_name + '.pth'
    print(model_path)

    
    state_dict = torch.load(model_path)
    model = SimpleCNN()
    model.load_state_dict(state_dict)

    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    outputs = []

    # Run model on data
    for i, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)

        with torch.no_grad():
            output = model(images)
            output = output.squeeze(0)

        outputs.append(output)
        print(f"Batch {i + 1}/{len(dataloader)}")

    # Convert the list of outputs to a tensor
    outputs = torch.cat(outputs, dim=0)                   


    return outputs

if __name__ == "__main__":
    # Construct the path to the data
    data_dir = os.path.join(os.getcwd(), 'data/raw/batch1')

    #data_dir = 'C:/Users/farim/Desktop/particle_tracking_02456/data/raw/batch1'
    data_loader = load_dataset_real(data_dir, 1)
    outputs = predict_model("Simple_CNN", data_loader)

    print(outputs.shape)

    #plot the output of the first image
    plt.figure()
    plt.imshow(outputs[0].numpy(), cmap='gray')    