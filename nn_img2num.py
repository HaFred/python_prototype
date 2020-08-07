import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import time



class NnImg2Num:
  def __init__(self):
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(784, 500)
            self.fc2 = nn.Linear(500,100)
            self.fc3 = nn.Linear(100, 10)
    
        def forward(self, x):
            x = x.view(-1, 784)
            #print('input shape is',x.shape)
            #print('self.fc1(x) shape is',self.fc1(x).shape)
            x = F.sigmoid(self.fc1(x))
            x = F.sigmoid(self.fc2(x))
            x = F.sigmoid(self.fc3(x))
            #print(x)
            return x
    
    self.model = Net()
    self.learning_rate=0.1
    self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
    self.numEpochs=200



  def forward(self,img):
    out=self.model.forward(Variable(img))
    return out


  def train(self):

### DATA LOADING:
    kwargs = {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=64, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=1000, shuffle=True, **kwargs)
    
    
### TRAINING:    
    def train(epoch):
        print('training epoch:', epoch)
        #time_meas=torch.Tensor(self.numEpochs)
        #start_time=time.clock()
        correct=0
        #self.model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            target_onehot=torch.zeros(target.shape[0],10)
            for i in range(target.shape[0]): target_onehot[i,int(target[i])]=1
           # print(target)
            data, target_onehot = Variable(data), Variable(target_onehot)
            self.optimizer.zero_grad()
            output = self.model.forward(data)
            #print('output shape is',output.shape)
            loss = F.mse_loss(output, target_onehot)
            loss.backward()
            self.optimizer.step()
            '''
            if batch_idx % 5 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.data[0]))
            '''
            pred= output.data.max(1, keepdim=True)[1] # get the index of the max sigmoid case
            correct += pred.eq(target.view_as(pred)).sum()
        acc= correct*100/len(train_loader.dataset)
        print ('Accuracy and loss are', acc, loss.data[0])
        return [loss.data[0],acc]


### TESTING:   
    def test():
        #self.model.eval()
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            target_onehot=torch.zeros(target.shape[0],10)
            for i in range(target.shape[0]): target_onehot[i,int(target[i])]=1
            data, target_onehot = Variable(data, volatile=True), Variable(target_onehot)
            output = self.model.forward(data)
           # print('output shape is',output.data.shape)
           # print('target shape is',output.data.shape)
            test_loss += F.mse_loss(output, target_onehot, size_average=False).data[0] # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max sigmoid case
            correct += pred.eq(target.view_as(pred)).cpu().sum()
    
        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    
    
    start_time=time.clock()
    time_meas=torch.Tensor(self.numEpochs)
    acc_train=torch.Tensor(self.numEpochs)
    error_train=torch.Tensor(self.numEpochs)
    for epoch in range(1, self.numEpochs + 1):
        [error_train[epoch-1],acc_train[epoch-1]]=train(epoch)
        print('seconds elapsed is ', time.clock()- start_time)
        time_meas[epoch-1] = time.clock()- start_time
    test()
    print('Here is your time to train',time_meas)
    print('Here is your train error',error_train)
    print('Here is your accuracy',acc_train)
