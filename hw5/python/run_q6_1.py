import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.optim import lr_scheduler


import torch.nn as nn
import torch.nn.functional as F

''' CNN on CIFAR dataset '''

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential (
            # 32-3+1 = 30*30*32
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # 15*15*32
            nn.MaxPool2d(2,2)
        ) 

        self.conv2 = nn.Sequential (
            # 15-3+1 = 13*13*64
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.conv3 = nn.Sequential (
            # 13-4+1=10*10*128
            nn.Conv2d(in_channels=64,out_channels=128, kernel_size=4),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.05),
            # 5*5*128 
            nn.MaxPool2d(2,2)
        )

        self.fc = nn.Sequential(
            # Fully connected: 3200->1024
            nn.Linear(3200, 1024),
            nn.ReLU(inplace=True),
            # Fully connected: 1024->512
            nn.Linear(1024,512),
            nn.ReLU(inplace=True),
            # Fully connected: 512->128
            nn.Linear(512,128),
            nn.Dropout(p=0.05),
            # Fully connected: 128->10
            nn.Linear(128,10)
        )

    # CNN Forward pass
    def forward(self, x):
        x = x.cuda()
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.shape[0],-1)
        x = self.fc(x)
        return x

PATH = './p2_model.pkl'

def main():
    # Hyperparameters 
    batch_size, num_epoch, lr = 64, 15, 0.001

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)
    # load and transform dataset
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = CNN()
    net.to(device)
   
    # Cross Entropy Loss
    criterion = nn.CrossEntropyLoss().to(device)
    # Adam Optimizer
    optimizer = optim.Adam(net.parameters(), lr=lr)
    # Reduce LR on plateau
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    for epoch in range(num_epoch):  # loop over the dataset multiple times
        for param_group in optimizer.param_groups:
            print("Epoch {}, learning rate: {}".format(epoch, param_group['lr']))
        scheduler.step(epoch)

        running_loss = 0.0
        
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 500 == 499:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')

    torch.save(net.state_dict(), PATH)


    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))

           
if __name__ == "__main__":
    main()