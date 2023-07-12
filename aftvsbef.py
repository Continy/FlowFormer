import numpy as np
import torch
import glob


def cal_EPE(gt_flow, est_flow):
    return torch.norm(gt_flow - est_flow, 2, 1).mean()


finetuned_path = 'C:/Users/zihaozhang/Desktop/git/FlowFormer-Official/results/tartanair/things'
things_path = 'C:/Users/zihaozhang/Desktop/git/FlowFormer-Official/results/tartanair/small/things'
gt_path = 'C:/Users/zihaozhang/Desktop/git/FlowFormer-Official/datasets/abandonedfactory/Easy/P001/flow'
finetuned_flow = sorted(glob.glob(finetuned_path + '/*.npy'))
things_flow = sorted(glob.glob(things_path + '/*.npy'))
gt_flow = sorted(glob.glob(gt_path + '/*.npy'))
finetuned_EPE = []
things_EPE = []
for i in range(len(finetuned_flow)):
    finetuned_flow_array = np.load(finetuned_flow[i])
    things_flow_array = np.load(things_flow[i])
    gt_flow_array = np.load(gt_flow[i])
    finetuned_EPE.append(
        cal_EPE(torch.from_numpy(gt_flow_array),
                torch.from_numpy(finetuned_flow_array)))
    things_EPE.append(
        cal_EPE(torch.from_numpy(gt_flow_array),
                torch.from_numpy(things_flow_array)))
print('finetuned EPE: {}'.format(np.mean(finetuned_EPE)))
print('things EPE: {}'.format(np.mean(things_EPE)))
