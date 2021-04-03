import os
import cv2
import time
import math
import numpy as np
import matplotlib as plt
from py_libseek import *
from PIL import Image

# Torch Imports
import torch
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Dataset
from rcnn import *

RES = (154, 154)
SEEK_SAVE = r"C:\Users\lja\OneDrive\Documents\uoft\Winter 2021\ECE496\repo\libseek-thermal\build\examples\Debug\seek_save.exe"


def save_image(path, file_name):
    pass


def read_tmp():
    os.system("\"{}\"".format(SEEK_SAVE) + " test.csv")
    return np.loadtxt(fname=open("test.csv"), dtype=np.uint32, delimiter=",")


def read_rgb(cam):
    ret, frame = cam.read()
    assert ret
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def main():

    print("Waiting for camera Initialization...")
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if cam.isOpened():
        print("Camera Initialized!")

    # Initialize NN forward prop.
    vgg = models.vgg16(pretrained=True)
    data_transform = transforms.Compose(
        [transforms.CenterCrop(154), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    avgpool = nn.AdaptiveAvgPool2d((7, 7))
    lstm_model = lstmModel(1024 * 7 * 7, 2048, 7)

    # image read
    for i in range(10):
        cam.read()

    rgb_img = read_rgb(cam)
    tmp_dat = read_tmp()

    rgb_dim, rgb_minmax = sample_img_dim(rgb_img)
    tmp_dim, tmp_minmax = sample_img_dim(tmp_dat)

    feature_queue = []

    while True:
        # Capture Image
        rgb_img = read_rgb(cam)
        tmp_dat = read_tmp()

        # Crop Image
        rgb_cropped = crop_and_resize_image(rgb_img, rgb_minmax, rgb_dim, RES)
        tmp_cropped = crop_and_resize_data(tmp_dat, tmp_minmax)
        print(rgb_cropped.shape)
        print(tmp_cropped.shape)

        # Feed Forward
        # Transform images
        rgb_cropped = Image.fromarray(rgb_cropped.astype(np.uint8))
        tmp_cropped = Image.fromarray(tmp_cropped.astype(np.uint8)).convert('RGB')
        rgb_cropped = data_transform(rgb_cropped)
        tmp_cropped = data_transform(tmp_cropped)
        rgb_feature = vgg.features(rgb_cropped)
        tmp_feature = vgg.features(tmp_cropped)

        # Concatenate features
        rgb_feature_tensor = torch.from_numpy(rgb_feature.detach().numpy())
        tmp_feature_tensor = torch.from_numpy(tmp_feature.detach().numpy())
        rgb_tmp_combined_tensor = torch.cat((rgb_feature_tensor, tmp_feature_tensor), dim=1)
        rgb_tmp_combined_tensor = avgpool(rgb_tmp_combined_tensor)
        rgb_tmp_combined_tensor = torch.flatten(rgb_tmp_combined_tensor)

        # Maintain Feature Queue
        feature_queue.append(rgb_tmp_combined_tensor.detach().numpy())
        if len(feature_queue) < 8:
            continue

        feature_input_tensor = torch.Tensor(np.array(feature_queue, ndmin=3))
        outputs = lstm_model(feature_input_tensor)
        print(outputs)
        feature_queue.pop(0)

        # Display Image
        fig, axs = plt.subplots(1, 2)

        axs[0].set_title("RGB Image")
        axs[0].imshow(rgb_cropped)

        axs[1].set_title("Temperature Image")
        axs[1].imshow(tmp_cropped)
        plt.show()



    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()