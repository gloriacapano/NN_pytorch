# XOR function if
#x1 or x2 is 1 than XOR = 1
# otherwise XOR = 0

# NN using Linear + ReLU

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
    def __init__(self, n_hidden):
        super(Net, self).__init__()
        self.n_hidden = n_hidden
        self.fc1 = nn.Linear(2, n_hidden)  # 2 Input nodes, 10 in hidden num_layers
        self.fc2 = nn.Linear(n_hidden,1) # 10 nodes in 1 hidden layer, 2 output nodes
        self.rl1 = nn.ReLU()
        self.rl2 = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.rl1(x)
        x = self.fc2(x)
        x = self.rl2(x)

        return x



import time
import math
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m*60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def trainIters(net, X, n_epochs, print_every = 1000, plot_every = 100, lr=0.001, momentum=0.1):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # reset
    plot_loss_total = 0

    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.1)

    loss_file = open('XOR_losses.txt', 'w')
    for epoch in range(n_epochs):
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
            print_loss_total += loss
            plot_loss_total += loss
            loss.backward()
            optimizer.step()

            if (epoch-1) % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (timeSince(start, epoch / n_epochs),epoch, epoch / n_epochs *100, print_loss_avg))

            if (epoch-1) % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0
                loss_file.write(str(plot_loss_avg)+'\n')
    showLoss(plot_losses)
    loss_file.close()
    return plot_losses


import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np

def showLoss(points):
    fig, ax = plt.subplots()
    plt.plot(points)
    plt.savefig("XOR_losses.png")


import os
nodes_hidden = [2,4, 6, 8, 10, 20, 50]
for nodes in nodes_hidden:
    net = Net(nodes)
    plot_loss = trainIters(net, X, 20000, print_every=1000, plot_every=100)
    filename1 = "XOR_losses_"+str(nodes)+".png"
    filename2 = "XOR_losses_"+str(nodes)+".txt"
    os.rename("XOR_losses.png", filename1)
    os.rename("XOR_losses.txt", filename2)
    print(net(Variable(torch.FloatTensor(X[0]))))


print('Finished Training')
