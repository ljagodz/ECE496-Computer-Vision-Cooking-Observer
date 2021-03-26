import os
import re
import torch
from torch import nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
from torchvision.io import read_image
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
        img_Image = Image.open(img_path)
        rgb_Image = Image.open(rgb_path)
        img_Image = img_Image.convert('RGB')
        rgb_Image = rgb_Image.convert('RGB')
        if self.transform is not None:
            img_Image = self.transform(img_Image)
            rgb_Image = self.transform(rgb_Image)

        return rgb_Image, img_Image, int(self.rgb_img_seq[idx].split('_')[-1][:-4])



def main():
    #PROJECT_FOLDER = './data/'
    #FEATURE_FOLDER = PROJECT_FOLDER + 'features'
    seq0 = ImageSequenceFolder('./data/dataset_by_run/dataset_by_run/0/')

if __name__ == "__main__":
    main()