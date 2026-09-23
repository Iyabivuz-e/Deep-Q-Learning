import torch
from torch import nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, states_dim, action_dim, hidden_layer_dim=16):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(states_dim, hidden_layer_dim) # Takes input states into hidden layer
        self.fc2 = nn.Linear(hidden_layer_dim, action_dim) # Takes hidden layer into output action layer
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
