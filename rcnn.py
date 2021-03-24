import os
#import cuda

# Torch Imports.
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms

# TODO:
# - may require a custom Dataset class implementation for our data.
# -- or use ImageFolder from datasets module of torchvision.
# - Combine vgg16 with RNN to be able to feed image sequences.
# -- need to read some papers.

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
def rcnnBuild(num_classes):

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

    # Define RNN classifier.
    rnn = nn.RNN(input_size=(512 * 7 * 7), hidden_size=4096, num_layers=2, nonlinearity='relu')

    # May need to connect nn.Linear layer at the output of rnn_classifier.

    # Define loss and optimizer.
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(vgg16.parameters(), lr=1e-3)

    return vgg16, rnn, loss, optimizer


if __name__ == "__main__":
    classes = [
        'No Pancake',
        'Raw',
        'Ready to Flip',
        'Bottom Up',
        'Ready to Remove',
        'Burnt'
    ]

    classes_dict = {key: item for key, item in enumerate(classes)}

    vgg16_model = rcnnBuild(len(classes))

    # Insert everything else here.