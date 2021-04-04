import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

import os
import time
import math

import numpy as np
from skimage import io
from skimage import data, color, img_as_ubyte
from skimage.feature import canny
from skimage.transform import hough_ellipse, resize
from skimage.draw import ellipse_perimeter

TEMP_CONST = 0.0363691


def sample_minmax(img_rgb):
    fig, ax = plt.subplots(1, 1)
    # img_rgb = io.imread(img_path)
    plt.imshow(img_rgb)

    while True:
        pts = []
        while len(pts) < 4:
            plt.title("Select 4 Points (Y-min, Y-max, X-min, X-max), in no particular order")
            pts = np.asarray(plt.ginput(4, timeout=60))
            if len(pts) < 4:
                plt.title("Too few points, starting over")
                time.sleep(1)

        y_max = max([p[1] for p in pts])
        y_min = min([p[1] for p in pts])
        x_max = max([p[0] for p in pts])
        x_min = min([p[0] for p in pts])

        d = 2
        phs = [plt.fill([x_min, x_min, x_max, x_max], [y_min, y_min+d, y_min+d, y_min], "w", lw=2),
               plt.fill([x_min, x_min, x_max, x_max], [y_max+d, y_max, y_max, y_max+d], "w", lw=2),
               plt.fill([x_min, x_min, x_min+d, x_min+d], [y_min, y_max, y_max, y_min], "w", lw=2),
               plt.fill([x_max-d, x_max-d, x_max, x_max], [y_min, y_max, y_max, y_min], "w", lw=2),
               ]

        plt.title("Happy?: KEY CLICK for YES - MOUSE CLICK for NO")
        fig.canvas.draw()

        if plt.waitforbuttonpress():
            break

        for ph in phs:
            for p in ph:
                p.remove()

    plt.close(fig)

    return x_max, x_min, y_max, y_min


# Crop, Resize RGB Image Data and Save to Desired File
def crop_and_resize_image(img, minmax, square_dim, final_dim):
    # Load Up Image
    img_crop = img[minmax['min']['y']: minmax['max']['y'], minmax['min']['x']:minmax['max']['x']]

    # Flatten Image
    img_square = resize(img_crop, square_dim, preserve_range=True, anti_aliasing=False).astype('uint8')

    # Scale Image
    img_scaled = resize(img_square, final_dim, preserve_range=True, anti_aliasing=False).astype('uint8')

    # Save Image
    # io.imsave(fname=out_path, arr=img_scaled)

    return img_scaled


# Crop Temperature Data and Save to Desired File
def crop_and_resize_data(csv_data, minmax):
    # Load Up CSV
    #csv_data = np.loadtxt(fname=open(csv_path), dtype=np.uint32, delimiter=",")

    # extract min and max x lengths.
    x_min = minmax['min']['x']
    x_max = minmax['max']['x']

    x_mid = int(math.floor((x_max - x_min)/2)) + x_min

    if x_mid < 77:
        x_mid = 77

    if x_mid > 129:
        x_mid = 129

    x_min = x_mid - 77
    x_max = x_mid + 77

    # Crop Data
    csv_data = csv_data[:, x_min:x_max]

    # Convert to Integers
    csv_data = csv_data.astype(int)

    # Flip (about x axis)
    csv_data = np.flip(m=csv_data, axis=(0, 1))

    # Create Image
    image = np.copy(csv_data)

    # Normalize
    # img_mean = np.mean(image)
    img_min = np.min(image)
    img_max = np.max(image)

    image = image - img_min

    image = image * (255/(img_max - img_min))

    image = np.clip(image, 0, 255)

    # Save Data
    # np.savetxt(fname=out_data, X=csv_data, delimiter=",")
    # io.imsave(fname=out_img, arr=image)

    return image


def sample_img_dim(img):
    # User Input
    x_max, x_min, y_max, y_min = sample_minmax(img)

    # Make sure MaxMin are Indices
    x_max = int(math.floor(x_max))
    x_min = int(math.floor(x_min))
    y_max = int(math.floor(y_max))
    y_min = int(math.floor(y_min))

    # Dimensions
    dim = (x_max - x_min, x_max - x_min)
    minmax = {"max": {"x": x_max, "y": y_max}, "min": {"x": x_min, "y": y_min}}

    return dim, minmax