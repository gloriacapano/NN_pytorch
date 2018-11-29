# XOR function if x1 or x2 is 1 than XOR = 1
# otherwise 0

# NN

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

# The four points
X = [[[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]]
Y = [[[0.0], [1.0], [1.0], [0.0]]]


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 10)  # 4 Input nodes, 10 in hidden num_layers
        self.fc2 = nn.Linear(10,1) # 10 nodes in 1 hidden layer, 2 output nodes
        self.rl1 = nn.ReLU()
        self.rl2 = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.rl1(x)
        x = self.fc2(x)
        x = self.rl2(x)

        return x



net = Net()


criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.1)

NumEpochs = 20000

for epoch in range(NumEpochs):
    running_loss = 0.0
    for i, data in enumerate(X, 0):
        inputs = data
        labels = Y[i]
        inputs = Variable(torch.FloatTensor(inputs))
        labels = Variable(torch.FloatTensor(labels))
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
    if epoch % 100 == 0:
        print('[%d, %5d] loss: %.3f' %
              (epoch, 4, running_loss/4))
        print(net(Variable(torch.FloatTensor(X[0,0]))))
        running_loss = 0.0
)

print('Finished Training')
print(net(Variable(torch.FloatTensor(X[0]))))
