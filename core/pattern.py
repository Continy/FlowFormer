import unittest
from datasets import TartanAir
from utils import frame_utils
import numpy as np

dataset = TartanAir(
    root=
    'C:/Users/zihaozhang/Desktop/git/FlowFormer-Official/datasets/abandonedfactory/Easy',
    folderlength=2)
print('find {} image pairs'.format(len(dataset.image_list)))
print('find {} flows'.format(len(dataset.flow_list)))
#print(dataset.image_list)
print(dataset.flow_list[1])
index = 1

img1 = frame_utils.read_gen(dataset.image_list[index][0])
img2 = frame_utils.read_gen(dataset.image_list[index][1])

flow = np.load(dataset.flow_list[index])
img1 = np.array(img1).astype(np.uint8)
img2 = np.array(img2).astype(np.uint8)
print(flow)