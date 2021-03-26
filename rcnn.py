import os
#import cuda

# Torch Imports.
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
import rcnn_data

# TODO:
# - may require a custom Dataset class implementation for our data.
# -- or use ImageFolder from datasets module of torchvision.
# - Combine vgg16 with RNN to be able to feed image sequences.
# -- need to read some papers.


class lstmModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(lstmModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.lstm = nn.LSTM(input_size, hidden_size)

        self.class_output = nn.Linear(hidden_size, num_classes)

    def forward(self, input):
        x = self.avgpool(input)
        x = torch.flatten(x, 1)
        x = self.lstm(x)
        x = self.output(x)
        return x


# Function to initialize CUDA stuff, provided using a machine with Nvidia GPU.
# This may not be needed, not sure.
def cudaSetup():
    cuda = torch.cuda('cuda');
    return cuda


def main():
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print("Using CUDA device.")



if __name__ == "__main__":
    classes = [
        'No Pancake',
        'Raw',
        'Ready to Flip',
        'Bottom Up',
        'Ready to Remove',
        'Burnt',
        'Obstruction'
    ]

    classes_dict = {key: item for key, item in enumerate(classes)}

    main()

    # Insert everything else here.