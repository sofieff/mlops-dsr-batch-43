import os
import wandb
from loadotenv import load_env
from torchvision.models import resnet18, ResNet
from torch import nn
from pathlib import Path
import torch
from torchvision.transforms import v2 as transforms

# this gives us access to the variables in .env file
#load_env()
wandb_api_key = os.environ.get("WANDB_API_KEY")

# This is the local folder where the wandb model will be downloaded
MODELS_DIR = "../models"
MODEL_FILENAME = "best_model.pth"

os.makedirs(MODELS_DIR, exist_ok=True)

def download_artifact():
    assert 'WANDB_API_KEY' in os.environ, "WANDB_API_KEY not found in environment variables"
    wandb.login(key=wandb_api_key)
    api = wandb.Api()
    
    wandb_org = os.environ.get("WANDB_ORG")
    wandb_project = os.environ.get("WANDB_PROJECT")
    wandb_model_name = os.environ.get("WANDB_MODEL_NAME")
    wandb_model_version = os.environ.get("WANDB_MODEL_VERSION")
    
    artifact_path = f'{wandb_org}/{wandb_project}/{wandb_model_name}:{wandb_model_version}'
    #print(f"Downloading artifact from {artifact_path}")
    
    artifact = api.artifact(artifact_path, type='model')
    artifact.download(root=MODELS_DIR)

#download_artifact()
def get_raw_model() -> str:
    ''' 
    Get the architecture of the model (random weights), this must match the architecture used during training
    '''
    architecture = resnet18(weights=None) # use random weights

    architecture.fc = nn.Sequential(
        nn.Linear(in_features=512, out_features=512),
        nn.ReLU(),
        nn.Linear(in_features=512, out_features=6)
    )
    return architecture

def load_model() -> ResNet:
    '''
    Load the model with the trained weights
    '''
    download_artifact()
    model = get_raw_model()

    # load the weights from the file into a state_dict
    model_state_dict_path = Path(MODELS_DIR) / MODEL_FILENAME
    model.state_dict = torch.load(model_state_dict_path, map_location=torch.device('cpu'))

    # merge weights into model architecture
    model.load_state_dict(model.state_dict, strict=True) 
    model.eval() # set model to evaluation mode - turn off Dropout and BatchNorm uses stats from training - IMPORTANT: must do this before inference

    return model

#live_resnet = load_model()
#print(live_resnet)


# we explicitely write the trnasforms with clear and ordered steps and sizes
def load_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(256), # resize to 256
            transforms.CenterCrop(224), # crop to 224x224 about the center
            transforms.ToImage(), # convert to image    
            transforms.ToDtype(torch.float, scale=True), # convert to float tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225]) # normalize to imagenet stats
        ]
    )