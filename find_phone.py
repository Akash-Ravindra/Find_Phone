import cv2 as cv
import numpy
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch import nn
from torch.nn.modules.conv import Conv2d
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import random
import os
import time
from tqdm import tqdm
torch.manual_seed(0)



def get_inputs():
    parser = argparse.ArgumentParser(description='Find Phone',usage='\t\t\tpython %(prog)s -f ./find_phone/51.jpg',)
    parser.add_argument('-f', type=str,nargs='?', default="./find_phone/51.jpg", help="The Path containing the photos and labels")
    
    
    args = parser.parse_args()
    path = args.f
    
    return [path]

class network_1v2(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
          Conv2d(3,16,5),
          nn.ReLU(),
          nn.MaxPool2d(2,2),
          Conv2d(16,32,5),
          nn.ReLU(),
          nn.MaxPool2d(2,2),
        )
        self.dense = nn.Sequential(
          nn.Linear(32*29*29,10000),
          nn.ReLU(),
          nn.Linear(10000,1000),
          nn.ReLU(),
          nn.Linear(1000,100),
          nn.ReLU(),
          nn.Linear(100,2)
        )
        
    def forward(self, x):
      output = self.dense(torch.flatten(self.cnn(x),1))
      return output


if __name__=="__main__":
    path = get_inputs()
    model = torch.load('./modelv2.pth')
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = model.to(device)
    transformer = transforms.Compose([
                transforms.Resize((128,128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
    img1 = Image.open(path[0])
    img = transformer(img1)
    output = model(img.unsqueeze(0).to(device)).cpu().detach().numpy()
    print(output)