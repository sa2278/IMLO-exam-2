import os
from tempfile import TemporaryDirectory
import torch.utils
import torch.utils.data
from torchsummary import summary
import torch 
import numpy as np 
from torchvision import datasets, transforms 
from torchvision import datasets
from torch import nn 
import torch.nn.functional as Func
import torch.optim as optim   
from torch.utils.tensorboard import SummaryWriter   
import pickle 



import argparse
import os
import tempfile

import ray
from ray import train, tune 
from ray.tune.schedulers import ASHAScheduler
from filelock import FileLock 
from ray.train import Checkpoint
from ray.tune.schedulers import AsyncHyperBandScheduler

writer = SummaryWriter() 

batch_size = 64 
model_pkl_file = "flowers102CNN.pt" 

trainTransform = transforms.Compose([transforms.AutoAugment(), transforms.ToTensor(), transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]), transforms.Resize((265,265)), transforms.CenterCrop((224,224)), transforms.RandomHorizontalFlip(0.5),transforms.RandomVerticalFlip(0.5), transforms.RandomRotation(degrees=(0, 45))]) 
testTransform = transforms.Compose([transforms.ToTensor(), transforms.Resize((224,224))]) 


def deviceType():
    deviceFlag = torch.device('cpu')
    if torch.cuda.is_available():
        deviceFlag = torch.device('cuda:0') 
    return deviceFlag 



trainset = datasets.Flowers102(root='./data', split= "train", download=True, transform=trainTransform)
train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

testset = datasets.Flowers102(root='./data', split="test", download=True, transform=testTransform)
test_dataloader = torch.utils.data.DataLoader(testset, batch_size=11, shuffle=True, num_workers=0) 

validationset = datasets.Flowers102(root='./data', split="val", download=True, transform=testTransform)
validation_dataloader = torch.utils.data.DataLoader(validationset, batch_size=16, shuffle=True, num_workers=0) 

singletestset = datasets.Flowers102(root='./data', split="test", download=True, transform=testTransform)
single_test_dataloader = torch.utils.data.DataLoader(singletestset, batch_size=1, shuffle=True, num_workers=0) 

class ResBottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels,stride = 1, downsample = None):
        super().__init__()
        self.downsample = downsample
        self.conv1 = nn.Conv2d(in_channels, out_channels//4, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(out_channels//4, out_channels//4, kernel_size=3, stride=2 if downsample else 1, padding=1)
        self.conv3 = nn.Conv2d(out_channels//4, out_channels, kernel_size=1, stride=1)
        self.shortcut = nn.Sequential()
        
        if self.downsample or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2 if self.downsample else 1),
                nn.BatchNorm2d(out_channels)
            )

        self.bn1 = nn.BatchNorm2d(out_channels//4)
        self.bn2 = nn.BatchNorm2d(out_channels//4)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        shortcut = self.shortcut(input)
        input = nn.ReLU()(self.bn1(self.conv1(input)))
        input = nn.ReLU()(self.bn2(self.conv2(input)))
        input = nn.ReLU()(self.bn3(self.conv3(input)))
        input = input + shortcut
        return nn.ReLU()(input)

class ResNet(nn.Module):
    def __init__(self, in_channels, resblock, repeat, useBottleneck=True, outputs=102):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        if useBottleneck:
            filters = [64, 256, 512, 1024, 2048]
        else:
            filters = [64, 64, 128, 256, 512]

        self.layer1 = nn.Sequential()
        self.layer1.add_module('conv2_1', resblock(filters[0], filters[1], downsample=False))
        for i in range(1, repeat[0]):
                self.layer1.add_module('conv2_%d'%(i+1,), resblock(filters[1], filters[1], downsample=False))

        self.layer2 = nn.Sequential()
        self.layer2.add_module('conv3_1', resblock(filters[1], filters[2], downsample=True))
        for i in range(1, repeat[1]):
                self.layer2.add_module('conv3_%d' % (i+1,), resblock(filters[2], filters[2], downsample=False))

        self.layer3 = nn.Sequential()
        self.layer3.add_module('conv4_1', resblock(filters[2], filters[3], downsample=True))
        for i in range(1, repeat[2]):
            self.layer3.add_module('conv2_%d' % (i+1,), resblock(filters[3], filters[3], downsample=False))

        self.layer4 = nn.Sequential()
        self.layer4.add_module('conv5_1', resblock(filters[3], filters[4], downsample=True))
        for i in range(1, repeat[3]):
            self.layer4.add_module('conv3_%d'%(i+1,), resblock(filters[4], filters[4], downsample=False))

        self.gap = nn.AdaptiveAvgPool2d(1)  
        self.fc1 = nn.Linear(filters[4], (1024))  
        self.dropout = nn.Dropout(0.5) 
        self.fc2 = nn.Linear(1024, 102) 
    


    def forward(self, input):
        input = self.layer0(input)
        input = self.layer1(input)
        input = self.layer2(input)
        input = self.layer3(input)
        input = self.layer4(input)
        input = self.gap(input)
        input = torch.flatten(input, start_dim=1)
        input = self.fc1(input)   

        input = self.dropout(input) 
        input = self.fc2(input)

        return input



net = ResNet(3, ResBottleneckBlock, [3, 4, 6, 3], useBottleneck=True, outputs=102)  
if torch.cuda.is_available():
   net = net.to(deviceType()) 

criterion = nn.CrossEntropyLoss() 
max_lr = 0.02 
weight_decay = 0.0001 
optimizer = optim.Adam(net.parameters(),lr = 0.001, weight_decay=weight_decay)  
# optimizer = optim.SGD(net.parameters(),lr =  0.001, weight_decay=weight_decay, momentum=0.9)


def train_model(epochs):  
    criterion = nn.CrossEntropyLoss() 
    # here the optimizer used is the stochastic gradient descent 
    for epoch in range(epochs):  # loop over the dataset multiple times  

        iterations = 0
        runningLoss = 0.0  
        trainAccuracy = 0  
        net.train() 
       

        for inputs, labels in iter(train_dataloader):
            # get the inputs; data is a list of [inputs, labels] and sends them to either cuda or cpu
            inputs = inputs.to(deviceType())
            labels = labels.to(deviceType())  
            outputs = net.forward(inputs) 
            loss = criterion(outputs, labels)     

            display = f'Epoch: {epoch + 1}/{epochs}, itrs: {iterations}, ' 
            display += f'Loss: {loss}'
            print(display)
            _, predicted = torch.max(outputs.data, 1)
            
            # zero the parameter gradients
            optimizer.zero_grad()  
            loss.backward()
            optimizer.step() 
            torch.cuda.empty_cache()

            runningLoss += loss.item()    
            trainAccuracy += (labels.data == predicted).sum().item() 
        writer.add_scalar("Loss/train", loss, epoch) 

        valRunningLoss = 0
        valAccuracy = 0   
        valAccuracyPercent = 0
        count = 0   
        with torch.no_grad():
            for images, labels in iter(validation_dataloader):
                
                images = images.to(deviceType())
                labels = labels.to(deviceType())
                
                valOutput = net.forward(images)
                _, predicted = torch.max(valOutput.data, 1)
                
                valRunningLoss += criterion(valOutput, labels).item()  
                count += labels.size(0)
                valAccuracy += (labels.data == predicted).sum().item() 
            
        print("the number correct is: ", valAccuracy,  " out of :", count)
        valAccuracyPercent = 100 * valAccuracy / count

        validationLoss = valRunningLoss / len(validation_dataloader)    
        display = f'Epoch: {epoch + 1}/{epochs}, itrs: {iterations}, '
        display += f'Validation loss: {round(validationLoss, 4)}, '
        display += f'Validation Accuracy: {round(valAccuracyPercent, 4)}%'
        print(display)
        
        runningLoss = 0
        iterations += 1   
        writer.add_scalar("Accuracy/train", valAccuracyPercent, epoch)   
  
    display2 = f'Finished training, '   
    display2 += f'Final training accuracy: {round(valAccuracyPercent, 4)}%, '
    display2 += f'Final training loss: {loss}, '
    print(display2)   


def testing_model(): 
    testAccuracy = 0  
    runningLoss = 0 
    count = batch_size 
    criterion = nn.CrossEntropyLoss() 
    for images, labels in (test_dataloader):
        
        images = images.to(deviceType())
        labels = labels.to(deviceType())
        
        testOutput = net(images)
        _, predicted = torch.max(testOutput.data, 1)
        
        runningLoss += criterion(testOutput, labels).item()  
        count += labels.size(0)
        testAccuracy += (labels.data == predicted).sum().item()               
    print(testAccuracy) 
    
    
    testLoss = runningLoss / count    
    testAccuracy = 100 * testAccuracy / count
    display = f'testing loss: {round(testLoss, 4)}, '
    display += f'testing Accuracy: {round(testAccuracy, 4)}%'
    print(display)  

def use_Net(image):  
    image.to(deviceType()) 
    output = net(image)
    return output.max(dim = 1)[1]

dataiter = iter(single_test_dataloader)
imagesLocal, labelsLocal = next(dataiter)   
imagesLocal = imagesLocal.to(deviceType())
labelsLocal = labelsLocal.to(deviceType())  
summary(net, input_size=(3, 256, 256))
writer.add_graph(net, imagesLocal)  

if __name__ == "__main__":
    print(deviceType())
    print("enter 1 to build from scratch, 2 to use prebuilt model") 
    n = int(input()) 
    if n == 1:     
        train_model(500)    
        torch.save(net.state_dict(), model_pkl_file) 
        net.eval()
        testing_model()  
    elif n == 2:  
        net = ResNet(3, ResBottleneckBlock, [3, 4, 6, 3], useBottleneck=True, outputs=102)   
        net.load_state_dict(torch.load(model_pkl_file, weights_only=True))
        net.eval()
        net.to(deviceType())
        testing_model()    
    elif n == 3:  
        net = ResNet(3, ResBottleneckBlock, [3, 4, 6, 3], useBottleneck=True, outputs=102)   
        net.load_state_dict(torch.load(model_pkl_file, weights_only=True))
        net.eval()
        net.to(deviceType()) 
        torch.save(net.state_dict(), "flowers102CNN.pth") 
        testing_model()    



output = use_Net(imagesLocal)
print(output, " and " ,labelsLocal.data) 


writer.flush() 
