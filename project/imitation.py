import numpy as np
import torch
import torch.optim as optim
import random
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn as nn

from RL import RLagent
import RL
from simulator import SJF, JobSet

class SJFdata(Dataset):
    def __init__(self, x, y=None):
        self.x = x
        # label is required to be a LongTensor
        self.y = y
        if y is not None:
            self.y = torch.LongTensor(y)
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        X = self.x[index]
        if self.y is not None:
            Y = self.y[index]
            return X, Y
        else:
            return X

agent = RLagent()
np.random.seed()
x_data = []
y_data = []

for i in range(500):
    j = JobSet(SJF)
    while not j.isEmpty():
        j.increaseTime()
        idx, invalid = j.pickNext(0)
        if invalid:
            continue
        else:
            y_data.append(idx)
            x_data.append(agent.state_to_image(j.getState()))


#shuffle
data = list(zip(x_data, y_data))
random.shuffle(data)
x_data, y_data = zip(*data)
data = SJFdata(x_data, y_data)

## split
train_set_size = int(len(data) * 0.9)
valid_set_size = len(data) - train_set_size
train_set, val_set = random_split(data, [train_set_size, valid_set_size])

## dataloader
batch_size = 10
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

model = agent.policy
loss = nn.CrossEntropyLoss() 
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
#optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
num_epoch = 10

for epoch in range(num_epoch):
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0

    model.train() 
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        train_pred = torch.FloatTensor()
        
        for k in range(len(data[0])):
            train_pred = torch.cat((train_pred, model(data[0][k])))
        batch_loss = loss(train_pred, data[1]) 
        batch_loss.backward()
        optimizer.step()
        train_acc += np.sum(np.argmax(train_pred.data.numpy(), axis=1) == data[1].numpy())
        train_loss += batch_loss.item()
    
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            
            val_pred = torch.FloatTensor()
            for k in range(len(data[0])):
                val_pred = torch.cat((val_pred, model(data[0][k])))

            batch_loss = loss(val_pred, data[1])

            val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
            val_loss += batch_loss.item()

        print('[%03d/%03d] Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % \
            (epoch + 1, num_epoch, \
             train_acc/train_set.__len__(), train_loss/train_set.__len__(), val_acc/val_set.__len__(), val_loss/val_set.__len__()))
        
        #f = open('imitation_model_' + str(epoch) + '.pkl', 'wb')
        torch.save(agent.policy, '/Users/mcnlab/Downloads/pcs/imitation_model_' + str(epoch))
        #f.close()
