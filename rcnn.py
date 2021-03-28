import os
#import cuda

# Torch Imports.
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from torchvision import datasets, models, transforms
import rcnn_data
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt

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

        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

        self.class_output = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        #out = self.avgpool(x)
        #out = torch.flatten(out, 2)
        out, (h_n, c_n) = self.lstm(x)
        out = out[:,-1,:]
        out = self.class_output(torch.relu(out))
        return out


def get_data_loader(batch_size, sequence_length, feature_path):
    # feature_path = './data/dataset_by_run/dataset_by_run/features_by_run/0/'
    features_dataset = rcnn_data.FeatureSequenceFolder(feature_path, sequence_length)

    indices = [i for i in range(len(features_dataset))]

    np.random.seed(1000)
    np.random.shuffle(indices)

    split_train = int(len(indices) * 0.8)
    split_val = int(len(indices) * 0.9)

    train_indices = indices[:split_train]
    val_indices = indices[split_train:split_val]
    test_indices = indices[split_val:]

    # Training
    train_sampler = SubsetRandomSampler(train_indices)
    train_loader = DataLoader(features_dataset, batch_size=batch_size, sampler=train_sampler)

    # Validation
    val_sampler = SubsetRandomSampler(val_indices)
    val_loader = DataLoader(features_dataset, batch_size=batch_size, sampler=val_sampler)

    # Test
    test_sampler = SubsetRandomSampler(test_indices)
    test_loader = DataLoader(features_dataset, batch_size=batch_size, sampler=test_sampler)

    return train_loader, val_loader, test_loader


def evaluate(net, loader, criterion, use_cuda):
    total_loss = 0.0
    total_acc = 0.0
    total_epoch = 0
    i = 0

    for inputs, labels in iter(loader):
        if use_cuda:
            inputs = inputs.cuda()
            labels = labels.cuda()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        pred = outputs.max(1, keepdim=True)[1]
        total_acc += pred.eq(labels.view_as(pred)).sum().item()

        total_epoch += len(labels)
        i += 1

    acc = float(total_acc) / total_epoch
    loss = float(total_loss) / (i + 1)
    return acc, loss


def main():
    # use_gpu = torch.cuda.is_available()
    # if use_gpu:
    #     print("Using CUDA device.")

    # Initialize for training.

    use_cuda = False #torch.cuda.is_available()

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

    # Hyperparamete declarations.
    batch_size = 1
    num_epochs = 15
    lr = 0.01
    momentum = 0.9

    # LSTM model.
    lstm_model = lstmModel(1024 * 7 * 7, 2048, num_classes)

    feature_path = './data/dataset_by_run/dataset_by_run/features_by_run/0/'
    train_loader, val_loader, test_loader = get_data_loader(1, 3, feature_path)
    #data, target = next(iter(train_loader))
    #print(lstm_model(data))


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(lstm_model.parameters(), lr, momentum)

    train_acc = np.zeros(num_epochs)
    train_loss = np.zeros(num_epochs)
    val_acc = np.zeros(num_epochs)
    val_loss = np.zeros(num_epochs)

    for epoch in range(num_epochs):
        i = 0
        for inputs, labels in iter(train_loader):
            if use_cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()

            outputs = lstm_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            print('Iteration %d Complete' % i)
            i += 1

        train_acc[epoch], train_loss[epoch] = evaluate(lstm_model, train_loader, criterion)
        val_acc[epoch], val_loss[epoch] = evaluate(lstm_model, val_loader, criterion)

    epochs = np.arange(1, num_epochs + 1)
    plt.title("Training Curve")
    plt.plot(epochs, train_loss, label="Train")
    plt.plot(epochs, val_loss, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.legend(loc='best')
    plt.show()

    plt.title("Training Curve")
    plt.plot(epochs, train_acc, label="Train")
    plt.plot(epochs, val_acc, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Training Accuracy")
    plt.legend(loc='best')
    plt.show()


if __name__ == "__main__":
    main()