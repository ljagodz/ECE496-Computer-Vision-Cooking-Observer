import os
#import cuda

# Torch Imports.
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms

#class pancakeDataset(Dataset):


use_gpu = torch.cuda.is_available()
if use_gpu:
    print("Using CUDA device.")


# Function to initialize CUDA stuff, provided using a machine with Nvidia GPU.
# This may not be needed, not sure.
def cudaSetup():
    cuda = torch.cuda('cuda');
    return cuda


def dataSetup():
    pass


# Build CNN (Currently making use of VGG16 architecture).
def cnnBuild(num_classes):

    # Define VGG16 model.
    vgg16 = models.vgg16(pretrained=True)

    # Modify to output num_classes instead of 1000.
    # https://www.kaggle.com/carloalbertobarbano/vgg16-transfer-learning-pytorch
    # START
    for param in vgg16.features.parameters():
        param.require_grad = False
    num_features = vgg16.classifier[6].in_features
    features = list(vgg16.classifier.children())[:-1]
    features.extend([nn.Linear(num_features, num_classes)])
    vgg16.classifier = nn.Sequential(*features)
    # END

    # Move model to GPU.
    if use_gpu:
        vgg16.cuda()

    # Define loss and optimizer.
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(vgg16.parameters(), lr=1e-3)

    return vgg16, loss, optimizer


if __name__ == "__main__":
    classes = [
        'empty',
        'raw'
    ]

    vgg16_model = cnnBuild(len(classes))

    # Insert everything else here.