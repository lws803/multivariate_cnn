import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

NUM_CLASSES = 4
WINDOW_SIZE = 3
GENERALISE = True
NUM_CONV_LAYERS = 2

X = np.array([[10, 20, 30], [70, 60, 50], [70, 70, 70], [10, 30, 10], [20, 20, 20]])  # Time series
y = np.array([0, 1, 2, 3, 2])  # Classifications

# TODO: Consider using padding instead. Always set WINDOW_SIZE to the largest window
# Smaller dataset will be padded instead

# TODO: Test it on the earthquake dataset and see if we can yield better results

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Network (nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        # Observations: Increasing the number of hidden layers post-conv increases the accuracy in prediction
        # in_channels determine the number of data points per sequence
        self.conv = nn.Conv1d(in_channels=WINDOW_SIZE, out_channels=10, kernel_size=2, padding=(2 // 2))
        self.conv2 = nn.Conv1d(in_channels=10, out_channels=10, kernel_size=2, padding=(2 // 2))
        self.bn1 = nn.BatchNorm1d(num_features=10)  # TODO: See if normalisation helps

        self.dropout = nn.Dropout(p=0.5)
        self.pool = nn.MaxPool1d(NUM_CONV_LAYERS)
        self.flatten = Flatten()
        self.linear1 = nn.Linear(10, 50)
        self.output = nn.Linear(50, NUM_CLASSES)

    def forward(self, x):
        x = self.conv(x)
        # x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        # x = self.bn1(x)
        x = F.relu(x)

        if GENERALISE:
            x = self.dropout(x)  # For added benefit of generalisation but will make training performance worse
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

for epoch in tqdm(range(1000)):  # loop over the dataset multiple times
    running_loss = 0.0

    optimizer.zero_grad()
    outputs = net(inputs)

    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    running_loss += loss.item()
    # print('[%d] loss: %.3f' %
    #     (epoch + 1, running_loss / (epoch + 1)))

print("====================Finished training=================")

evaluation_set = [
    [15, 25, 35],
    [45, 70, 80],
    [60, 60, 60],
    [20, 35, 20]
]

for item in evaluation_set:
    print(item)
    x_input = np.array(item)
    x_input = x_input.reshape((1, 3, 1))
    test_input = torch.tensor(x_input)
    test_input = test_input.type(torch.FloatTensor)
    outputs = net(test_input)
    conf, predicted = torch.max(outputs.data, 1)
    print("Predicted class:", int(predicted[0]), "confidence:", float(conf))
