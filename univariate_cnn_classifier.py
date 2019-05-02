import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

X = np.array([[10,20,30], [50,60,70], [70, 60, 50]])
y = np.array([0,1,2])


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Network (nn.Module):
    def __init__ (self):
        super(Network, self).__init__()
        # Observations: Increasing the number of hidden layers post-conv increases the accuracy in prediction
        # in_channels determine the number of data points per sequence
        # TODO: Tune and find the best model
        self.conv = nn.Conv1d(in_channels=3, out_channels=10, kernel_size=2,padding=(2 // 2))
        self.conv2 = nn.Conv1d(in_channels=10, out_channels=10, kernel_size=2,padding=(2 // 2))
        self.pool = nn.MaxPool1d(2)
        self.flatten = Flatten()
        self.linear1 = nn.Linear(10, 50)
        self.output = nn.Linear(50, 3)

    def forward (self, x):
        x = self.conv(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        # TODO: Consider adding a dropout layer here
        x = self.pool(x)
        x = self.flatten(x)
        x = F.relu(self.linear1(x))
        x = F.softmax(self.output(x))
        return x

net = Network()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())
X = X.reshape((X.shape[0], X.shape[1], 1))

inputs = torch.tensor(X)
labels = torch.tensor(y)
inputs = inputs.type(torch.FloatTensor)

for epoch in range(1000):  # loop over the dataset multiple times
    running_loss = 0.0

    optimizer.zero_grad()
    outputs = net(inputs)
     
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    running_loss += loss.item()
    print('[%d] loss: %.3f' %
        (epoch + 1, running_loss / (epoch+1)))

print ("Finished training")


x_input = np.array([15, 25, 35])
x_input = x_input.reshape((1, 3, 1))
test_input = torch.tensor(x_input)
test_input = test_input.type(torch.FloatTensor)
outputs = net(test_input)
conf, predicted = torch.max(outputs.data, 1)
print (predicted[0], conf)

x_input = np.array([50, 60, 70])
x_input = x_input.reshape((1, 3, 1))
test_input = torch.tensor(x_input)
test_input = test_input.type(torch.FloatTensor)
outputs = net(test_input)
conf, predicted = torch.max(outputs.data, 1)
print (predicted[0], conf)

x_input = np.array([75, 60, 50])
x_input = x_input.reshape((1, 3, 1))
test_input = torch.tensor(x_input)
test_input = test_input.type(torch.FloatTensor)
outputs = net(test_input)
conf, predicted = torch.max(outputs.data, 1)
print (predicted[0], conf)

