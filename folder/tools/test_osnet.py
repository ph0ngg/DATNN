import torch

state_dict = torch.load('D:\PhongNghiem\\folder\\tools\khongconv5_lossview0,5.pth')
print(state_dict['model'].keys())