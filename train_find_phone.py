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
    parser = argparse.ArgumentParser(description='Find Phone Trainer',usage='\t\t\tpython %(prog)s -f "~/find_phone/"',)
    parser.add_argument('-f', type=str,nargs='?', default="./find_phone/", help="The Path containing the photos and labels")
    
    
    args = parser.parse_args()
    path = args.f
    
    return [path]

def read_labels(path):
    images_dict = {}
    i=0
    with open(path+"labels.txt") as f:
        for line in f:
            (path, val1, val2) = line.split()
            images_dict[i] = [str(path),float(val1), float(val2)]
            i+=1

    unique_ids = list(images_dict.keys())
    train_ids = unique_ids[:]
    return images_dict, train_ids

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

class find_phone_dataset(torch.utils.data.Dataset):

  def __init__(self, split="train", images_dict={}, ids=[]):
    self.split = split
    self.images_dict = images_dict
    self.ids = ids

  def __getitem__(self, index):
    transformer = transforms.Compose([
            transforms.Resize((128,128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    id = self.ids[index]
    img1 = Image.open('./Find_Phone/find_phone/'+self.images_dict[id][0])
    img = transformer(img1)
    label = torch.FloatTensor(self.images_dict[id][1:])
    return img, label

  def __len__(self):
    return len(self.ids)

def train(model,device,train_loader,optimizer,epochs,loss_func):
  loss_values = []
  for epoch in tqdm(range(epochs)):  # loop over the dataset multiple times
      running_loss = 0.0
      for i, (images, labels) in enumerate(train_loader):
          # get the inputs; data is a list of [inputs, labels]
          images = images.to(device)
          labels = labels.to(device)
          labels = torch.autograd.Variable(labels.float()) 

          # zero the parameter gradients
          optimizer.zero_grad()

          # forward + backward + optimize
          outputs = model(images)
          loss = loss_func(outputs, labels)
          loss.backward()
          optimizer.step()

          # print statistics
          running_loss += loss.item()
          accuracy = (1 - running_loss/len(train_dataset))*100
          # if (i+1) % batch_size == 0:
          #       print('Train Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {}'.format(epoch+1, epochs, i+1, len(train_loader), loss.item(), accuracy))
      running_loss /= len(train_loader)
      loss_values.append(running_loss)
  print('Finished Training Loss = ',)
  plt.plot(loss_values)
  return running_loss


if __name__=="__main__":
    path = get_inputs()
    img_dict, train_ids = read_labels(path[0])
    batch_size = 10
    train_dataset = find_phone_dataset(images_dict=img_dict, ids=train_ids)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    start_time = time.time()
    num_epoch = 20
        
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    modelv2 = network_1v2().to(device)

    # You can start your code here
    #################################################
    optimizer = torch.optim.Adam(modelv2.parameters(), lr = 1e-3)
    loss_func = nn.MSELoss()
    train(modelv2,device, train_dataloader,optimizer,num_epoch,loss_func)
    #################################################

    end_time = time.time()
    elasped_time = end_time-start_time
    torch.save(modelv2, './modelv2.pth')