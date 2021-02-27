import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque, namedtuple
import numpy as np
import random

BUFFER_SIZE = int(1e7)  # replay buffer size
BATCH_SIZE = 1024        # minibatch size
GAMMA = 0.99               # discount factor
LR = 1e-4               # learning rate 
UPDATE_EVERY = 5        # how often to update the network
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
class QNetwork(nn.Module):
    def __init__(self, input_shape, input_channels, action_size):
        super(QNetwork, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(input_channels, 32, kernel_size = 8, stride = 4, padding = 2), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size = 4, stride = 2, padding = 1), nn.ReLU())
 
        self.pool = nn.MaxPool2d(2 ,stride = 2)

        out_size = self.calc_size(input_channels, input_shape, self.conv1, self.conv2)
        self.fc = nn.Sequential(nn.Linear(out_size, 512), nn.ReLU())
        self.out = nn.Linear(512, action_size)
        
    def forward(self, state):
        state = self.conv1(state)
        state = self.pool(state)
        state = self.conv2(state)
        state = self.pool(state)
        #state = self.conv3(state)
        #state = self.pool(state)

        state = state.view(state.size(0), -1)
        state = self.fc(state)
        out = self.out(state)

        return out

    def calc_size(self, input_channels, input_shape, *layers):
        x = torch.rand(1,input_channels, input_shape[0], input_shape[1])
        for layer in layers:
            x = layer(x)
            #print(x.size())
            x = self.pool(x)
            #print(x.size())
        out_size = x.size(1)*x.size(2)*x.size(3)
        return out_size

class ReplayBuffer:

    def __init__(self, action_size, buffer_size, batch_size):

        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    
    def add_exp(self, state, action, reward, next_state, done):
        #state = np.array(state)
        #next_state = np.array(next_state)
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample_batch(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.Tensor(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.Tensor(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.Tensor(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.Tensor(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.Tensor(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)
    def __len__(self):
        return len(self.memory)
    
class Agent():

    def __init__(self, action_size, input_shape, input_channels):
    
        self.input_shape = input_shape
        self.action_size = action_size

        self.qnetwork_local = QNetwork(input_shape, input_channels, action_size).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE)
    
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
    
        self.memory.add_exp([state], action, reward, [next_state], done)
        
        self.t_step +=1
        if self.t_step % UPDATE_EVERY == 0:
            if len(self.memory) > BATCH_SIZE:
                
                experiences = self.memory.sample_batch()
                self.learn(experiences, GAMMA)

    def get_action(self, state, eps):
        # Epsilon-greedy action selection
        if random.random() > eps:
            state = torch.Tensor(state).float().unsqueeze(0).to(device)
            self.qnetwork_local.eval()
            with torch.no_grad():
                action_values = self.qnetwork_local(state)
            self.qnetwork_local.train()
            action = np.argmax(action_values.cpu().data.numpy())
            return action
        else:
            action = random.choice(range(self.action_size))
            return action
        

    def learn(self, experiences, gamma):
        states,actions, rewards, next_states, dones = experiences

        Q_targets_next, _ = torch.max(self.qnetwork_local(next_states).detach(), dim = 1, keepdim = True)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
    
        Q_expected = torch.gather(self.qnetwork_local(states), dim = 1, index = actions)
        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()