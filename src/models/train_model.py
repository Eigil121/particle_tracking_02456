import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
from src.data.particle_dataset import load_dataset as load_dataset_real
from src.data.simulate_dataset import load_dataset as load_dataset_sim 
from architectures import *
import argparse
import yaml

def load_dataset(batch_size, dataset_name):
    
    # Load dataset
    if dataset_name == 'simulate':
        data_loader = load_dataset_sim(batch_size)
    
    elif dataset_name == 'inference':
        data_loader = load_dataset_real(batch_size, dataset_type="inference")
    
    elif dataset_name == 'supervised':
        data_loader = load_dataset_real(batch_size, dataset_type="supervised")
    
    else:
        raise ValueError('dataset type not recognized')
    
    return data_loader

def load_model(model_name, architecture):
    try:
        model = eval(architecture + "()")

    except:
        raise ValueError('model architecture not recognized')

    if model_name + '.pth' in os.listdir('models'):
        state_dict = torch.load("models/" + model_name + '.pth')
        model.load_state_dict(state_dict)
        print(f"Model '{model_name}' loaded with architecture '{architecture}'")
    
    else:
        print(f"New model, '{model_name}', created with architecture '{architecture}'")

    return model

def save_model(model, model_name):
    save_path = os.path.join(os.getcwd(), 'models')
    final_model_path = os.path.join(save_path, f'{model_name}.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"Model '{model_name}' saved at {final_model_path}")

def train(model, dataloader, num_epochs=1, learning_rate=0.001):
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

        for i, (images, labels, image_info) in enumerate(dataloader):
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



def main():
    parser = argparse.ArgumentParser(description='Train the model.')
    parser.add_argument('config', type=str, help='Experiment name.')
    args = parser.parse_args()

    # Load the yaml configuration file
    experiment_configs = yaml.load(open("src/experiments/"+ args.config + ".yaml"), Loader=yaml.FullLoader)
    
    dataset_name = experiment_configs['data']['dataset']
    batch_size = experiment_configs['data']['batch_size']

    # Load dataset
    data_loader = load_dataset(batch_size, dataset_name)

    # Load model
    model = load_model(experiment_configs['model']['model_name'], experiment_configs['model']['architecture'])
    
    # Give kwargs to train function
    kwargs = {"num_epochs": experiment_configs['training']['epochs'], "learning_rate": experiment_configs['model']['learning_rate']}
    
    # Train model
    training_history, model = train(model, data_loader, **kwargs)
    
    if experiment_configs['training']['epochs'] > 2:
        # Plot training history
        plt.figure()
        plt.plot(training_history)
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.title("Training History")
        plt.show()

    # Save model
    if experiment_configs['model']['savename'] in [None, "None"]:
        save_name = experiment_configs['model']['model_name']
    else:
        save_name = experiment_configs['model']['savename']

    save_model(model, save_name)
    

    # Visualize output
    if experiment_configs['visualize']["plot"]:
        from src.visualization.visualize import visualize_output

        dataset_name = experiment_configs['visualize']['dataset']
        n_max = experiment_configs['visualize']['n_max']

        # Load dataset
        data_loader = load_dataset(n_max, dataset_name)

        sample_image, sample_mask, image_info = next(iter(data_loader))
        model.eval()

        with torch.no_grad():
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            sample_image = sample_image.to(device)
            output = model(sample_image)

        visualize_output(sample_image, sample_mask, output, image_info, nmax=n_max)


        

if __name__ == "__main__":
    main()