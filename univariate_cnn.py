import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)

raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
# choose a number of time steps
n_steps = 3
# split into samples
X, y = split_sequence(raw_seq, n_steps)

# summarize the data
for i in range(len(X)):
	print(X[i], y[i])


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Network (nn.Module):
    def __init__ (self):
        super(Network, self).__init__()
        # Observations: Increasing the number of hidden layers post-conv increases the accuracy in prediction
        self.conv = nn.Conv1d(in_channels=3, out_channels=150, kernel_size=2,padding=(2 // 2))
        self.pool = nn.MaxPool1d(2)
        self.flatten = Flatten()
        self.linear1 = nn.Linear(150, 50)
        self.linear2 = nn.Linear(50, 1)

    def forward (self, x):
        x = self.conv(x)
        x = self.pool(F.relu(x))
        x = self.flatten(x)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

net = Network()
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters())
X = X.reshape((X.shape[0], X.shape[1], 1))

inputs = torch.tensor(X)
labels = torch.tensor(y)
inputs,labels = inputs.type(torch.FloatTensor),labels.type(torch.FloatTensor)

for epoch in range(100):  # loop over the dataset multiple times
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


x_input = np.array([20, 30, 40])
x_input = x_input.reshape((1, 3, 1))
test_input = torch.tensor(x_input)
test_input = test_input.type(torch.FloatTensor)
print (net(test_input))

x_input = np.array([10, 20, 30])
x_input = x_input.reshape((1, 3, 1))
test_input = torch.tensor(x_input)
test_input = test_input.type(torch.FloatTensor)
print (net(test_input))

x_input = np.array([10, 40, 60])
x_input = x_input.reshape((1, 3, 1))
test_input = torch.tensor(x_input)
test_input = test_input.type(torch.FloatTensor)
print (net(test_input))
