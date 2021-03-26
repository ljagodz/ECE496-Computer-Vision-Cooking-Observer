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


def main():
    # use_gpu = torch.cuda.is_available()
    # if use_gpu:
    #     print("Using CUDA device.")

    # Initialize for training.
    classes = [
        'No Pancake',
        'Raw',
        'Ready to Flip',
        'Bottom Up',
        'Ready to Remove',
        'Burnt',
        'Obstruction'
    ]

    num_classes = len(classes)

    classes_dict = {key: item for key, item in enumerate(classes)}

    lstm_model = lstmModel(1024 * 7 * 7, 2048, num_classes)


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