# -*- coding: utf-8 -*-

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F

BATCH_SIZE = 4
EPOCH = 2

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
trainset = torchvision.datasets.CIFAR10(root='./data',train = True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset,batch_size = BATCH_SIZE,
                                          shuffle = True)

testset = torchvision.datasets.CIFAR10(root='./data',train = False,
                                        download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset,batch_size = BATCH_SIZE,
                                          shuffle = False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class CNN_NET(torch.nn.Module):
    def __init__(self):
        super(CNN_NET,self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels = 3,out_channels = 64,kernel_size = 5)
        self.pool = torch.nn.MaxPool2d(kernel_size = 3,stride = 2)
        self.conv2 = torch.nn.Conv2d(64,64,5)
        self.fc1 = torch.nn.Linear(64*4*4,384)
        self.fc2 = torch.nn.Linear(384,192)
        self.fc3 = torch.nn.Linear(192,10)

    def forward(self,x):
        x=self.pool(F.relu(self.conv1(x)))
        x=self.pool(F.relu(self.conv2(x)))
        x=x.view(-1,64*4*4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = CNN_NET()

import torch.optim as optim

optimizer = optim.SGD(net.parameters(),lr=0.001,momentum=0.9)
loss_func =torch.nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    running_loss = 0.0
    for step, data in enumerate(trainloader):
        b_x,b_y=data
        outputs = net.forward(b_x)
        loss = loss_func(outputs, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if step % 1000 == 999:    
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, step + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

dataiter = iter(trainloader)
images, labels = dataiter.next()
images_comb = torchvision.utils.make_grid(images)
images_comb_unnor = (images_comb*0.5+0.5).numpy()
plt.imshow(np.transpose(images_comb_unnor, (1, 2, 0)))
plt.show()

predicts=net.forward(images)



correct = 0
total = 0
with torch.no_grad():
    for (images,labels) in testloader:
        outputs = net(images)
        numbers,predicted = torch.max(outputs.data,1)
        total +=labels.size(0)
        correct+=(predicted==labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
