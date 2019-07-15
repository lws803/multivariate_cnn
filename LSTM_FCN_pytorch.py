
# coding: utf-8

# In[1]:


get_ipython().system('curl https://course-v3.fast.ai/setup/colab | bash')


# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import pathlib
import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt

from tqdm import tqdm

from fastai import *


# In[4]:


get_ipython().system('curl -O http://www.timeseriesclassification.com/Downloads/Earthquakes.zip')


# In[5]:


get_ipython().system('unzip Earthquakes.zip')


# In[ ]:


bs = 64


# In[ ]:


DATASET = 'Earthquakes'
classes = 2


# In[ ]:


path = pathlib.Path('')


# In[ ]:


def one_hot_encode(input, labels):
    m = input.shape[0]
    output = np.zeros((m, labels), dtype=int)
    row_index = np.arange(m)
    output[row_index, input] = 1
    return output

def split_xy(data, classes):
    X = data[:, 1:]
    y = data[:, 0].astype(int)
    # hot encode
    #y = one_hot_encode(y, classes)
    return X, y

def create_dataset(X, y, device):
    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y, dtype=torch.long, device=device)
    return TensorDataset(X_tensor, y_tensor)

def load_data(path, classes):
    data = np.loadtxt(path)
    return split_xy(data, classes)
    


# The outputs of the model should be of size (minibatch, C). On the other hand the target `y` should contain the indices of the classes.

# In[ ]:


# load training dataset
X_train, y_train = load_data(path/'Earthquakes_TRAIN.txt', classes) 

# load testing dataset
X_test, y_test = load_data(path/'Earthquakes_TEST.txt', classes)


# In[11]:


print('X_train %s   y_train %s' % (X_train.shape, y_train.shape))
print('X_test  %s   y_test  %s' % (X_test.shape, y_test.shape))


# As the classes are imbalanced, get the count for each class, to use later in the sampling

# In[12]:


class_0_count = (y_train==0).sum()
class_1_count = (y_train==1).sum()

class_0_count, class_1_count


# load the numpy training and test sets into pytorch Dataset object

# In[ ]:


cuda = torch.device('cuda')     # Default CUDA device


# In[ ]:


train_ds = create_dataset(X_train, y_train, cuda)
test_ds  = create_dataset(X_test, y_test, cuda)


# pass the Dataset objects into a DataLoader

# In[15]:


class_sample_count = [class_0_count, class_1_count] # dataset has 10 class-1 samples, 1 class-2 samples, etc.
weights = 1 / torch.Tensor(class_sample_count)
sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, bs)


# In[ ]:


train_dl = DataLoader(train_ds, batch_size=bs, shuffle=False)#, sampler = sampler)
test_dl = DataLoader(test_ds, batch_size=bs, shuffle=False)


# ## LSTM-FCN
# ### LSMT block
# A shuffle layer + LSTM layer + Dropout layer

# In[ ]:


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


# ### FCN block
# 
# #### Convolutional block

# In[ ]:


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


# #### FCN block

# In[ ]:


class BlockFCN(nn.Module):
    def __init__(self, time_steps, channels=[1, 128, 256, 128], kernels=[8, 5, 3], mom=0.99, eps=0.001):
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


# ### LSTM-FCN

# In[ ]:


class LSTMFCN(nn.Module):
    def __init__(self, time_steps, num_variables=1, lstm_hs=256, channels=[1, 128, 256, 128]):
        super().__init__()
        self.lstm_block = BlockLSTM(time_steps, 1, lstm_hs)
        self.fcn_block = BlockFCN(time_steps)
        self.dense = nn.Linear(channels[-1] + lstm_hs, num_variables)
        self.softmax = nn.LogSoftmax(dim=1) #nn.Softmax(dim=1)
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


# ## Training

# In[21]:


time_steps = X_train.shape[1]
num_variables = classes

time_steps, num_variables


# In[ ]:


model = LSTMFCN(time_steps, num_variables).cuda()


# look at the different blocks of the Model

# In[23]:


# model summary
for m in model.children():
    print(m.training)#, m)
    for j in m.children():
        print(j.training, j)


# look at the parameters (i.e. weights) in each layer

# In[24]:


[p.shape for p in model.parameters()]


# Define a learner class to automate the learning process

# In[ ]:


class SimpleLearner():
    def __init__(self, data, model, loss_func, wd = 1e-5):
        self.data, self.model, self.loss_func = data, model, loss_func
        self.wd = wd
    
    def update_manualgrad(self, x,y,lr):
        y_hat = self.model(x)
        # weight decay
        w2 = 0.
        for p in model.parameters(): w2 += (p**2).sum()
        # add to regular loss
        loss = self.loss_func(y_hat, y) + w2 * self.wd
        loss.backward()
        with torch.no_grad():
            for p in model.parameters():
                p.sub_(lr * p.grad)
                p.grad.zero_()
        return loss.item()

    def update(self, x,y,lr):
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
            for x,y in self.data[0]:
                current_loss = self.update(x, y , lr)
                losses.append(current_loss)
        return losses
    
    def evaluate(self, X):
        """Evaluate the given data loader on the model and return predictions"""
        result = None
        for x, y in X:
            y_hat = self.model(x)
            result = y_hat if result is None else np.concatenate((result, y_hat), axis=0)
        return result


# In[ ]:


model = LSTMFCN(time_steps, num_variables).cuda()


# train the model using the DataLoader

# In[ ]:


# depending on the number of classes, use a Binary Cross Entropy or a Negative Log Likelihood loss for more than two classes
loss_func = nn.NLLLoss().cuda() # weight=weights
acc_func = accuracy_thresh


# In[73]:


lr = 2e-2
learner = SimpleLearner([train_dl, test_dl], model, loss_func)
losses = learner.fit(10)


# In[74]:


plt.plot(losses)


# In[ ]:


y_pred = learner.evaluate(test_dl)


# In[82]:


((y_test - y_pred.argmax(axis=1))**2).mean()


# #### Training with fastai

# In[ ]:


model = LSTMFCN(time_steps, num_variables).cuda()


# In[ ]:


data = DataBunch(train_dl=train_dl, valid_dl=test_dl, path=path)
learner = Learner(data, model, loss_func=loss_func, metrics=accuracy)


# In[85]:


learner.unfreeze()
learner.lr_find()
learner.recorder.plot()


# In[86]:


learner.fit(10, lr=3e-3)


# In[89]:


learner.get_preds(DatasetType.Valid)


# In[90]:


learner.get_preds(DatasetType.Train)


# #### Training one cycle

# In[ ]:


data = DataBunch(train_dl=train_dl, valid_dl=test_dl, path=path)
learner = Learner(data, model, loss_func=loss_func, metrics=accuracy)
learner.fit(10, lr=5e-5)


# In[ ]:


model.eval()
model.train()

