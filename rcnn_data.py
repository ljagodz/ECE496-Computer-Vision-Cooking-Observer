import os
import re
import torch
from torch import nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
from torchvision.io import read_image
from torchvision.datasets.folder import default_loader
from PIL import Image


class ImageSequenceFolder(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.img_dir = data_dir + 'img/'
        self.rgb_dir = data_dir + 'rgb/'
        self.transform = transform

        # init the img sequence.
        img_dir_list = os.listdir(self.img_dir)
        self.length = len(img_dir_list)
        temp_img_list = [[file.split('_'), file] for file in img_dir_list]
        img_dict = {int(file[0][0]): file[-1] for file in temp_img_list}
        self.rgb_img_seq = []
        for i in range(self.length):
            self.rgb_img_seq.append(img_dict[i])

        # init the rgb sequence.
        # rgb_dir_list = os.listdir(rgb_dir)
        # temp_rgb_list = [[file.split('_'), file] for file in rgb_dir_list]
        # rgb_dict = {int(file[0][0]): file[-1] for file in temp_rgb_list}
        # self.rgb_seq = []
        # for i in range(self.length):
        #     self.rgb_seq.append(rgb_dict[i])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.rgb_img_seq[idx])
        rgb_path = os.path.join(self.rgb_dir, self.rgb_img_seq[idx])
        img_Image = default_loader(img_path)
        rgb_Image = default_loader(rgb_path)
        if self.transform:
            img_Image = self.transform(img_Image)
            rgb_Image = self.transform(rgb_Image)

        return rgb_Image, img_Image, int(self.rgb_img_seq[idx].split('_')[-1][:-4]), str(self.rgb_img_seq[idx][:-4])


# https://github.com/ashwinhprasad/PyTorch-For-DeepLearning/blob/master/RNN/RNNs.ipynb
class FeatureSequenceFolder(Dataset):
    def __init__(self, data_dir, seq_len, loader=torch.load):
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.loader = loader

        # Initialize sequences from feature tensors list.
        feature_dir_list = os.listdir(self.data_dir)
        temp_feature_list = [[file.split('_'), file] for file in feature_dir_list]
        feature_dict = {int(file[0][0]): file[-1] for file in temp_feature_list}
        self.feature_seq_list = []
        self.feature_label_list = []
        for i in range(len(feature_dir_list) - self.seq_len):
            temp_list = []
            for j in range(i, i + self.seq_len):
                temp_list.append(feature_dict[j])
            self.feature_seq_list.append(temp_list)
            self.feature_label_list.append(int(temp_list[-1].split('_')[-1][:-7]))

        self.length = len(self.feature_label_list)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        feature_file_sequence = self.feature_seq_list[idx]
        feature_label = self.feature_label_list[idx]
        feature_sequence = []
        for i in range(self.seq_len):
            feature_path = os.path.join(self.data_dir, feature_file_sequence[i])
            feature_tensor = self.loader(feature_path)
            feature_sequence.append(feature_tensor)

        return feature_sequence, feature_label

# def main():
#     #PROJECT_FOLDER = './data/'
#     #FEATURE_FOLDER = PROJECT_FOLDER + 'features'
#     seq0 = ImageSequenceFolder('./data/dataset_by_run/dataset_by_run/0/')
#
# if __name__ == "__main__":
#     main()