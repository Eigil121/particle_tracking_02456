import click
import torch

from src.models.architectures import SimpleCNN
@click.command()
@click.argument('model_name', type=str)
def predict_model(model_name):
    print(f"Predicting with {model_name}...")
    model_path = 'models/' + model_name + '.pth'
    print(model_path)
    model_data = torch.load(model_path)
    
    loaded_model = SimpleCNN()

    loaded_model.load_state_dict(model_data.load_state_dict())

    loaded_model.eval()

    print(loaded_model)
    # Print parameters
    for name, param in loaded_model.named_parameters():
        if param.requires_grad:
            print(name, param.data)

if __name__ == "__main__":
    
    predict_model()