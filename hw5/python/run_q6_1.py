import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import os
import pandas as pd

class ImageDataset(Dataset):
    def __init__(self, data, data_type='train'):
        if data_type == "train":
            self.X = data['train_data']
            self.y = data['train_labels']
        else:
            self.X = data['test_data']
            self.y = data['test_labels']

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        return x, y


import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.optim import lr_scheduler
import string 
from torch.utils.data import Dataset, DataLoader
import scipy
from scipy import io
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 2048)
        self.fc2 = nn.Linear(2048,4096)
        self.fc3 = nn.Linear(4096,1024)
        self.fc4 = nn.Linear(1024,256)
        self.fc5 = nn.Linear(256, output_dim)

        self.relu = nn.ReLU()
        # self.softmax = nn.Softmax(dim=1)
    


    # CNN Forward pass
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.fc5(x)
        x = self.relu(x)
        # out = self.softmax(x)
        return x

class CNN(nn.Module):
  # input: (B, 32,32)
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32,64, kernel_size=5)
        self.fc1 = nn.Linear(32*32, 256)
        self.fc2 = nn.Linear(256, 36)



    # CNN Forward pass
    def forward(self, x):
        B, M = x.shape
        x = x.view(B, 1,32,32) # C = 1

        x = F.relu(self.conv1(x))
        #x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv3(x),2))
        x = F.dropout(x, p=0.5, training=self.training)

        x = x.view(B, 32*32)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x



def main():
    # hyperparameters 
    batch_size, num_epoch, lr = 64, 50, 0.00005

    # set device 
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)

    # load train, test data 
    train_data = scipy.io.loadmat('./nist36_train.mat')
    test_data = scipy.io.loadmat('./nist36_test.mat')

    # train, test datasets
    train_set = ImageDataset(data=train_data, data_type='train')
    test_set = ImageDataset(data=test_data, data_type='test')
    train_size = len(train_set)
    test_size = len(test_set)

    print('train_size', train_size)
    print('test-size', test_size)
    # train, test data loaders 
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                            shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                            shuffle=False, num_workers=2)
    

    # classes 
    letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])

    # net = MLP(1024,36)
    net = CNN()
    net.to(device)
   
    # Cross Entropy Loss
    criterion = nn.CrossEntropyLoss().to(device)
    # Adam Optimizer
    optimizer = optim.Adam(net.parameters(), lr=lr)
    # Reduce LR on plateau
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5)

    avg_accs = []
    losses = []

    net.train()

    for epoch in range(num_epoch):  
        # for param_group in optimizer.param_groups:
        #     print("Epoch {}, learning rate: {}".format(epoch, param_group['lr']))
        scheduler.step(epoch)


        total_loss = 0.0
        batch_avg_accs = []
        for i, data in enumerate(train_loader):
            
            # get the inputs; data is a list of [inputs, labels]
            # X=(64,1024), y=(64,36)
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs.float())
            # labels = labels.cpu().detach().numpy()
            # outputs = outputs.cpu().detach().numpy()
            pred_label = letters[np.argmax(outputs.cpu().detach().numpy(), axis=1)]
            gt_label = letters[np.argmax(labels.cpu().detach().numpy(), axis=1)]
            # print(pred_label, gt_label)

            batch_avg_acc = np.count_nonzero(pred_label == gt_label) / batch_size
            batch_avg_accs.append(batch_avg_acc)


            # print(f'gt: {gt_label}, pred:{pred_label}')
            loss = criterion(outputs, torch.argmax(labels, 1))

            loss.backward()
            optimizer.step()

            # print statistics
            loss = loss.item()
            total_loss += loss # sum over all batches
            


        avg_acc = np.mean(batch_avg_accs)
        avg_accs.append(avg_acc)
        avg_loss = total_loss /  i # divide by num batches 
        losses.append(avg_loss)
        if epoch % 1 == 0:
            print(f'epoch:{epoch}, avg_loss:{avg_loss}, avg_acc:{avg_acc}')


    # Q5.3.1
    import matplotlib.pyplot as plt
    # visualize some results
    plt.figure()
    plt.plot(range(num_epoch), losses, '-r', label='loss')
    plt.show()

    plt.figure()
    plt.plot(range(num_epoch), avg_accs, '-b', label='accuracy')
    plt.show()
            

    print('Finished Training')
    PATH = f'model_ep{num_epoch}_bs{batch_size}_lr{lr}'
    torch.save(net.state_dict(), PATH)


    correct = 0
    total = 0
    net.eval()
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, axis=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: %d %%' % (
        100 * correct / total))

           
if __name__ == "__main__":
    main()