import os
import torch
from torchvision import transforms
import dataset
from tqdm.notebook import tqdm
from time import time

N_CHANNELS = 3

dataset = dataset.BreakHis('./dataset/BreaKHis_v1/', transform=transforms.ToTensor())
full_loader = torch.utils.data.DataLoader(dataset, shuffle=False, num_workers=8)

before = time()
mean = torch.zeros(3)
std = torch.zeros(3)
print('==> Computing mean and std..')
for inputs, _labels in full_loader:
    for i in range(N_CHANNELS):
        mean[i] += inputs[:,i,:,:].mean()
        std[i] += inputs[:,i,:,:].std()
mean.div_(len(dataset))
std.div_(len(dataset))
print(mean, std)

print("time elapsed: ", time()-before)