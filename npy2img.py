#transform npy to img
import sys

sys.path.append('core')
from utils import flow_viz
import numpy as np
import os
import glob
from PIL import Image

npy_path = 'datasets/abandonedfactory/Easy/P001/flow'
pattern = os.path.join(npy_path, '*.npy')
filelist = glob.glob(pattern)
save_path = 'datasets/abandonedfactory/Easy/P001/flow/'
#check if save_path exists
if not os.path.exists(save_path):
    os.makedirs(save_path)
for i in range(len(filelist)):
    flow = np.load(filelist[i])
    flow_img = flow_viz.flow_to_image(flow)
    image = Image.fromarray(flow_img)
    image.save(save_path + str(i).zfill(6) + '.png')
    print('已保存：{}/{}'.format(i + 1, len(filelist)))
