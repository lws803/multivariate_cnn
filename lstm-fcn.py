import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils import load_data, create_dataset
from torch.utils.data import DataLoader
from model import LstmFCN
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=64, type=int, help='batch size')
parser.add_argument('--classes', default=2, type=int, help='number of classifications')
parser.add_argument('--epochs', default=2000, type=int, help='number of epochs to run training')
parser.add_argument('--train', action='store_true', help='training mode')
parser.add_argument('--save_path', default="data/models/", type=str, help='model storage path')
parser.add_argument('--test_set', default="data/Earthquakes_TEST.txt", type=str, help='test data path')
parser.add_argument('--train_set', default="data/Earthquakes_TRAIN.txt", type=str, help='train data path')

args = parser.parse_args()

class Learner():
    def __init__(self, data, model, loss_func, wd=1e-5):
        self.data, self.model, self.loss_func = data, model, loss_func
        self.wd = wd

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
        torch.save(self.model.state_dict(), args.save_path + "model.pth")
        return losses

    def evaluate(self, X):
        """Evaluate the given data loader on the model and return predictions"""
        combined_array = np.array([], dtype=int)
        result = None

        for x, y in X:
            y_hat = self.model(x)
            conf, predicted = torch.max(y_hat.data, 1)
            predicted = list(predicted)
            result = y_hat.cpu().detach().numpy() \
                if result is None else np.concatenate((result, y_hat.cpu().detach().numpy()), axis=0)

            for i in range(0, len(predicted)):
                combined_array = np.append(combined_array, int(predicted[i]))
        print(combined_array)
        return result

def make_directories():
    if not os.path.exists("data/models"):
        os.makedirs("data/models")


if __name__ == "__main__":
    make_directories()

    X_train, y_train = load_data(args.train_set, args.classes)
    X_test, y_test = load_data(args.test_set, args.classes)
    print(y_test)
    print('X_train %s   y_train %s' % (X_train.shape, y_train.shape))

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    train_ds = create_dataset(X_train, y_train, device)
    test_ds = create_dataset(X_test, y_test, device)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    if torch.cuda.is_available():
        loss_func = nn.NLLLoss().cuda()
    else:
        loss_func = nn.NLLLoss().cpu()

    time_steps = X_train.shape[1]

    if torch.cuda.is_available():
        model = LstmFCN(time_steps, args.classes).cuda()
    else:
        model = LstmFCN(time_steps, args.classes).cpu()

    learner = Learner([train_dl, test_dl], model, loss_func)
    if not args.train:
        model.load_state_dict(torch.load(args.save_path + "model.pth"))

    if args.train:
        losses = learner.fit(args.epochs)
        # TODO: Plot the losses

    result = learner.evaluate(test_dl)
    print("loss:", ((y_test - result.argmax(axis=1))**2).mean())
