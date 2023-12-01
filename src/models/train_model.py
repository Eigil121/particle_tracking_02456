import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
from src.data.particle_dataset import load_dataset as load_dataset_real
from src.data.simulate_dataset import load_dataset as load_dataset_sim, generate_data 
from architectures import SimpleCNN, ImprovedCNN
import argparse
import yaml

def load_model(model_name, architecture):
    if architecture == 'SimpleCNN':
        model = SimpleCNN()
    elif architecture == 'ImprovedCNN':
        model = ImprovedCNN()
    else:
        raise ValueError('model architecture not recognized')

    if model_name + '.pth' in os.listdir('models'):
        state_dict = torch.load("models/" + model_name + '.pth')
        model.load_state_dict(state_dict)
        print(f"Model '{model_name}' loaded")
    
    else:
        print(f"New model, '{model_name}', created")

    return model

def save_model(model, model_name):
    save_path = os.path.join(os.getcwd(), 'models')
    final_model_path = os.path.join(save_path, f'{model_name}.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"Model '{model_name}' saved at {final_model_path}")

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


    return train_loss_history, model



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




def main():
    parser = argparse.ArgumentParser(description='Train the model.')
    parser.add_argument('config', type=str, help='Experiment name.')
    args = parser.parse_args()

    # Load the yaml configuration file
    experiment_configs = yaml.load(open("src/experiments/"+ args.config + ".yaml"), Loader=yaml.FullLoader)
    
    # Load dataset
    if experiment_configs['data']['dataset'] == 'simulate':
        data_loader = load_dataset_sim(experiment_configs['data']['batch_size'])
    
    elif experiment_configs['data']['dataset'] == 'inference':
        data_loader = load_dataset_real(experiment_configs['data']['batch_size'], dataset_type = "inference")
    
    elif experiment_configs['data']['dataset'] == 'supervised':
        data_loader = load_dataset_real(experiment_configs['data']['batch_size'], dataset_type = "supervised")
    
    else:
        raise ValueError('dataset type not recognized')

    # Load model
    model = load_model(experiment_configs['model']['model_name'], experiment_configs['model']['architecture'])
    
    # Give kwargs to train function
    kwargs = {"num_epochs": experiment_configs['training']['epochs'], "learning_rate": experiment_configs['model']['learning_rate']}
    
    # Train model
    training_history, model = train(model, data_loader, save_model = True, model_name='Simple_CNN', **kwargs)
    
    # Plot training history
    plt.figure()
    plt.plot(training_history)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Training History")
    plt.show()

    # Save model
    if experiment_configs['model']['save_name'] == None or experiment_configs['model']['savename'] == "None":
        save_name = experiment_configs['model']['model_name']
    else:
        save_name = experiment_configs['model']['savename']

    save_model(model, save_name)

    #visualize the model output for a new image
    sample_image, sample_mask = generate_data()
    
    visualize_output(model, sample_image, sample_mask)


if __name__ == "__main__":
    main()