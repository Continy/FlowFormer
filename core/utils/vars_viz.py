import torch
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import cv2


def read_flo_file(file_path):
    with open(file_path, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)[0]
            h = np.fromfile(f, np.int32, count=1)[0]
            data = np.fromfile(f, np.float32, count=2 * w * h)
            # Reshape data into 3D array (columns, rows, bands)
            data2D = np.resize(data, (h, w, 2))
            return data2D


def heatmap(data):

    #(1,H,W)-->(3,H,W)
    min, max = torch.min(data), torch.max(data)
    data = (data - min) / (max - min) * 255
    img = data.numpy().astype(np.uint8)
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    img = img[:, :, ::-1]
    return img


def colorbar(data):
    colors = [
        '#313695', '#4575b4', '#74add1', '#abd9e9', '#e0f3f8', '#ffffbf',
        '#fee090', '#fdae61', '#f46d43', '#d73027', '#a50026'
    ]
    cmap = mcolors.ListedColormap(colors)
    data[data > 1] = 1
    data = data.numpy()
    img = plt.imshow(data, cmap=cmap)
    return img


if __name__ == '__main__':
    data = torch.rand(480, 640)
    img = colorbar(data)
    plt.colorbar(img)
    plt.show()
