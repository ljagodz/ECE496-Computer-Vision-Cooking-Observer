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
# RGB_PATH = r"9\rgb"
# TMP_PATH = r"9\img"

# # IMG_PATH
# rgb_dir_list = os.listdir(RGB_PATH)
# length = len(rgb_dir_list)
# temp_img_list = [[file.split('_'), file] for file in rgb_dir_list]
# rgb_dict = {int(file[0][0]): file[-1] for file in temp_img_list}
#
# # TMP_PATH
# tmp_dir_list = os.listdir(TMP_PATH)
# length = len(tmp_dir_list)
# temp_img_list = [[file.split('_'), file] for file in tmp_dir_list]
# tmp_dict = {int(file[0][0]): file[-1] for file in temp_img_list}


def save_image(path, file_name):
    pass


def read_tmp():
    os.system("\"{}\"".format(SEEK_SAVE) + " test.csv")
    return np.loadtxt(fname=open("test.csv"), dtype=np.uint32, delimiter=",")
    # img = cv2.imread(TMP_PATH + r"\{}".format(tmp_dict[i]))
    # return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_rgb(cam):
    # image read
    for i in range(5):
        cam.read()

    ret, frame = cam.read()
    assert ret
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # img = cv2.imread(RGB_PATH + r"\{}".format(rgb_dict[i]))
    # return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_cameras(cam):
    # Start Temperature Capture
    os.system("\"{}\"".format(SEEK_SAVE) + " test.csv")

    # Read RGB Image
    rgb_img = read_rgb(cam)

    # Read TMP Data
    tmp_dat = np.loadtxt(fname=open("test.csv"), dtype=np.uint32, delimiter=",")

    return rgb_img, tmp_dat


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
    lstm_model_path = 'state.pt'
    lstm_model.load_state_dict(torch.load(lstm_model_path, torch.device('cpu')))
    classes = [
        'No Pancake',
        'Raw',
        'Ready to Flip',
        'Bottom Up',
        'Ready to Remove',
        'Burnt',
        'Obstruction'
    ]


    # ite = 0
    rgb_img, tmp_dat = read_cameras(cam)
    # ite += 1

    rgb_dim, rgb_minmax = sample_img_dim(rgb_img)
    tmp_dim, tmp_minmax = sample_img_dim(tmp_dat)
    plt.close('all')
    plt.pause(0.1)

    feature_queue = []

    # Display Image
    fig, axs = plt.subplots(1, 2)

    axs[0].set_title("RGB Image")
    ax0 = axs[0].imshow(np.zeros((154, 154, 3)))

    axs[1].set_title("Temperature Image")
    ax1 = axs[1].imshow(np.zeros((154, 154, 3)))
    fig.suptitle("Undefined")

    plt.ion()

    while True:
        # Capture Image
        rgb_img, tmp_dat = read_cameras(cam)
        # ite += 1

        # Crop Image
        rgb_cropped = crop_and_resize_image(rgb_img, rgb_minmax, rgb_dim, RES)
        #print(rgb_cropped.shape)
        tmp_cropped = crop_and_resize_data(tmp_dat, tmp_minmax)
        #print(tmp_cropped.shape)
        # print(rgb_cropped)
        # print(tmp_cropped)

        # Feed Forward
        # Transform images
        rgb_cropped_show = Image.fromarray(rgb_cropped.astype(np.uint8))
        print(rgb_cropped_show)
        tmp_cropped_show = Image.fromarray(tmp_cropped.astype(np.uint8)).convert('RGB')
        print(tmp_cropped_show)
        rgb_cropped = data_transform(rgb_cropped_show)
        tmp_cropped = data_transform(tmp_cropped_show)

        # Add 4th dimension
        rgb_cropped = torch.Tensor(rgb_cropped).unsqueeze(0)
        tmp_cropped = torch.Tensor(tmp_cropped).unsqueeze(0)

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
        # print(outputs)
        predicted_class = outputs.detach().numpy().argmax()
        print(classes[predicted_class])
        feature_queue.pop(0)

        #axs[0].set_title("RGB Image")
        ax0.set_data(rgb_cropped_show)

        #axs[1].set_title("Temperature Image")
        ax1.set_data(tmp_cropped_show)
        # plt.show()
        fig.suptitle(classes[predicted_class])
        plt.pause(0.01)

    plt.ioff()
    plt.show()
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()