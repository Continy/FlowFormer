import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2


def flow_var_to_img(var):

    #(1,H,W)-->(3,H,W)
    min, max = torch.min(var), torch.max(var)
    var = (var - min) / (max - min) * 255
    img = var.numpy().astype(np.uint8)
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    img = img[:, :, ::-1]
    return img


if __name__ == '__main__':
    x = torch.randn(1, 1, 480, 640)
    x.squeeze_(0)
    x.squeeze_(0)
    img = flow_var_to_img(x)
    print(img.shape)
    plt.imshow(img)
    plt.show()