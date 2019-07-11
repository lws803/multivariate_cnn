import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils import load_data, create_dataset
from torch.utils.data import DataLoader

batch_size = 64
classes = 2
epochs = 100
kernel_size = 8

class BlockLSTM(nn.Module):
    def __init__(self, time_steps, num_layers, lstm_hs, dropout=0.8, attention=False):
        super().__init__()
        self.lstm = nn.LSTM(input_size=time_steps, hidden_size=lstm_hs, num_layers=num_layers)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # input is of the form (batch_size, num_layers, time_steps), e.g. (128, 1, 512)
        x = torch.transpose(x, 0, 1)
        # lstm layer is of the form (num_layers, batch_size, time_steps)
        x, (h_n, c_n) = self.lstm(x)
        # dropout layer input shape (Sequence Length, Batch Size, Hidden Size * Num Directions)
        y = self.dropout(x)
        # output shape is same as Dropout intput
        return y


class BlockFCNConv(nn.Module):
    def __init__(self, in_channel=1, out_channel=128, kernel_size=8, momentum=0.99, epsilon=0.001, squeeze=False):
        super().__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size=kernel_size)
        self.batch_norm = nn.BatchNorm1d(num_features=out_channel, eps=epsilon, momentum=momentum)
        self.relu = nn.ReLU()

    def forward(self, x):
        # input (batch_size, num_variables, time_steps), e.g. (128, 1, 512)
        x = self.conv(x)
        # input (batch_size, out_channel, L_out)
        x = self.batch_norm(x)
        # same shape as input
        y = self.relu(x)
        return y

class BlockFCN(nn.Module):
    def __init__(self, time_steps, channels=[1, 128, 256, 128], kernels=[kernel_size, 5, 3], mom=0.99, eps=0.001):
        super().__init__()
        self.conv1 = BlockFCNConv(channels[0], channels[1], kernels[0], momentum=mom, epsilon=eps, squeeze=True)
        self.conv2 = BlockFCNConv(channels[1], channels[2], kernels[1], momentum=mom, epsilon=eps, squeeze=True)
        self.conv3 = BlockFCNConv(channels[2], channels[3], kernels[2], momentum=mom, epsilon=eps)
        output_size = time_steps - sum(kernels) + len(kernels)
        self.global_pooling = nn.AvgPool1d(kernel_size=output_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # apply Global Average Pooling 1D
        y = self.global_pooling(x)
        return y


class LSTMFCN(nn.Module):
    def __init__(self, time_steps, num_variables=1, lstm_hs=256, channels=[1, 128, 256, 128]):
        super().__init__()
        self.lstm_block = BlockLSTM(time_steps, 1, lstm_hs)
        self.fcn_block = BlockFCN(time_steps)
        self.dense = nn.Linear(channels[-1] + lstm_hs, num_variables)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # input is (batch_size, time_steps), it has to be (batch_size, 1, time_steps)
        x = x.unsqueeze(1)
        # pass input through LSTM block
        x1 = self.lstm_block(x)
        x1 = torch.squeeze(x1)
        # pass input through FCN block
        x2 = self.fcn_block(x)
        x2 = torch.squeeze(x2)
        # concatenate blocks output
        x = torch.cat([x1, x2], 1)
        # pass through Linear layer
        x = self.dense(x)
        #x = torch.squeeze(x)
        # pass through Softmax activation
        y = self.softmax(x)
        return y


# TODO: Try to understand the data type in this dataset

X_train, y_train = load_data('data/Earthquakes_TRAIN.txt', classes)
X_test, y_test = load_data('data/Earthquakes_TEST.txt', classes)
# X_train = np.array([[10, 20, 30], [70, 60, 50], [70, 70, 70], [10, 30, 10], [20, 20, 20]])  # Time series
# y_train = np.array([0, 1, 2, 3, 2])  # Classifications
# X_test = np.array([
#     [15, 25, 35],
#     [45, 70, 80],
#     [60, 60, 60]])  # Time series
# y_test = np.array([0, 0, 2])  # Classifications

print(y_test)
print('X_train %s   y_train %s' % (X_train.shape, y_train.shape))

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
train_ds = create_dataset(X_train, y_train, device)
test_ds = create_dataset(X_test, y_test, device)
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

if torch.cuda.is_available():
    loss_func = nn.NLLLoss().cuda()
else:
    loss_func = nn.NLLLoss().cpu()

time_steps = X_train.shape[1]

if torch.cuda.is_available():
    model = LSTMFCN(time_steps, classes).cuda()
else:
    model = LSTMFCN(time_steps, classes).cpu()


class SimpleLearner():
    def __init__(self, data, model, loss_func, wd=1e-5):
        self.data, self.model, self.loss_func = data, model, loss_func
        self.wd = wd

    def update_manualgrad(self, x, y, lr):
        y_hat = self.model(x)
        # weight decay
        w2 = 0.
        for p in model.parameters():
            w2 += (p**2).sum()
        # add to regular loss
        loss = self.loss_func(y_hat, y) + w2 * self.wd
        loss.backward()
        with torch.no_grad():
            for p in model.parameters():
                p.sub_(lr * p.grad)
                p.grad.zero_()
        return loss.item()

    def update(self, x, y, lr):
        opt = optim.Adam(self.model.parameters(), lr)
        y_hat = self.model(x)
        loss = self.loss_func(y_hat, y)
        loss.backward()
        opt.step()
        opt.zero_grad()
        return loss.item()

    def fit(self, epochs=1, lr=1e-3):
        """Train the model"""
        losses = []
        for i in tqdm(range(epochs)):
            for x, y in self.data[0]:
                current_loss = self.update(x, y, lr)
                losses.append(current_loss)
        return losses

    def evaluate(self, X):
        """Evaluate the given data loader on the model and return predictions"""
        for x, y in X:
            hits = 0
            misses = 0
            y_hat = self.model(x)
            conf, predicted = torch.max(y_hat.data, 1)
            predicted = list(predicted)
            correct_y = list(y)
            for i in range(0, len(predicted)):
                if predicted[i] == correct_y[i]:
                    hits += 1
                else:
                    misses += 1



learner = SimpleLearner([train_dl, test_dl], model, loss_func)
losses = learner.fit(epochs)
learner.evaluate(test_dl)
