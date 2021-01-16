#unknown parameters: Ti, M, epsilon greedy, (optimizer, learning_rate), gamma, iteration

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from torch.autograd import Variable
from torchvision import datasets, transforms, models

"""network"""
class policy(nn.Module):
    def __init__(self):
        super(policy, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, 5, 1, 2),
            nn.ReLU(),

            nn.AvgPool2d(2, stride=2),

            nn.Conv2d(8, 16, 5, 1, 2),
            nn.ReLU(),

            nn.AvgPool2d(2, stride=2),
        )
        self.fc = nn.Sequential(
            nn.Linear(5*56*16, 69),
            nn.Tanh(),

            nn.Linear(69, 11),
            nn.Softmax()
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)


"""agent"""
from torch.distributions import Categorical
import math
class RLagent():
  def __init__(self, train = True):
    self.policy = policy()
    self.epsilon = 0.7
    self.log = Variable(torch.Tensor())
    #self.optimizer = torch.optim.SGD(self.policy.parameters(), lr=0.1, momentum=0.9)
    self.optimizer = torch.optim.RMSprop(self.policy.parameters(), lr=0.001)
    # if train:
    #   self.policy.train()
    # else:
    #   self.policy.eval()
  def __state_to_image(self, state):
    state = np.expand_dims(state, 0)
    state = torch.from_numpy(state).float()
    #state = Variable(state)
    return state[None, ...]

  def refresh_traj(self):
    self.log = Variable(torch.FloatTensor())

  def select_action_eval(self, state):
    state = self.__state_to_image(state)
    #state = state.to('cuda')
    prob = self.policy(state)
    action = torch.topk(prob, 11).indices
    return action.tolist()
    #action = torch.argmax(prob)
    #return action.item()

  
  def select_action_train(self, state):
    state = self.__state_to_image(state)
    prob = self.policy(state)
    c = Categorical(prob)

    # epsilon-greedy, choose an action
    if random.random() > self.epsilon:
      action = torch.argmax(prob)
    else:
      action = torch.Tensor([random.randrange(11)]) # random or sample?

    # record the log probability of the chosen action
    self.log = torch.cat([self.log, c.log_prob(action)])
    return action.item()

'''
for using a model which is already trained:
  agent = RLagent(train = False)
  agent.policy = model
  action = select_action_eval(state)

which 'state' is a 2d numpy array, 'model is the model you've trained
select_action_eval() return the index of the picked job, from 0 to 9; could be 10 which means no picked job
'''