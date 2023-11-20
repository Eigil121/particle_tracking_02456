import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
#from torch.utils.data import 
from src.data.particle_dataset import load_dataset as load_dataset_real
from src.data.simulate_dataset import load_dataset as load_dataset_sim, generate_data 
from architectures import SimpleCNN, ParticleDetectionUNet



def train(model, dataloader, num_epochs=1, learning_rate=0.001, save_model = False, model_name='final_model'):
    # Move model to device (CPU/GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define loss function and optimizer
    #criterion = nn.BCEWithLogitsLoss()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    train_loss_history = []

    for epoch in range(num_epochs):
        model.train()

        for i, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)


            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            loss.backward()
            optimizer.step()

            # Print training information
            print(f"Epoch {epoch + 1}/{num_epochs}, Batch {i + 1}/{len(dataloader)}, Loss: {loss.item():.4f}")

            # Append current loss to history
            train_loss_history.append(loss.item())

    if save_model:
        # Save the final model
        save_path = os.path.join(os.getcwd(), 'models')
        final_model_path = os.path.join(save_path, f'{model_name}.pth')
        #torch.save({
        #'model_state_dict': model.state_dict(),
        #'model_architecture': architecture},
        #final_model_path)
        torch.save(model.state_dict(), final_model_path)
        print(f"Model '{model_name}' saved at {final_model_path}")

    return train_loss_history


def visualize_output(model, sample_image, sample_mask):
    model.eval()
    with torch.no_grad():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sample_image = sample_image.unsqueeze(0).to(device)
        output = model(sample_image)
        output = output.squeeze(0)

    sample_image = sample_image.squeeze(0)
    sample_image = sample_image.to('cpu')
    output = output.to('cpu')


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
    
    data_dir = 'particle_tracking_02456/data/'
    data_loader = load_dataset_sim(data_dir, 4)

    model = SimpleCNN()
    #model = ParticleDetectionUNet()
    train(model, data_loader, num_epochs=1, save_model = False, model_name='Simple_CNN')
    
    
    #visualize the model output for a new image
    sample_image, sample_mask = generate_data()
    
    visualize_output(model, sample_image, sample_mask)



