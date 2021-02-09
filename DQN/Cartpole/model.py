import torch
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple, deque
import numpy as np
import random
import torch.nn.functional as F

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 512        # minibatch size
GAMMA = 0.95            # discount factor
LR = 1e-3               # learning rate 
UPDATE_EVERY = 5        # how often to update the network
device = 'cuda' if torch.cuda.is_available() else 'cpu'
TAU = 1e-3

class QNetwork(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(input_shape,24), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(24,24), nn.ReLU())
        self.out = nn.Sequential(nn.Linear(24,output_shape))
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.out(x)
        return x

class ReplayBuffer:

    def __init__(self, buffer_size, batch_size):

        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    
    def add_exp(self, state, action, reward, next_state, done):
        state = np.array(state)
        next_state = np.array(next_state)
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

    def __init__(self, input_shape, action_size):
    
        self.input_shape = input_shape
        self.action_size = action_size

        self.qnetwork_local = QNetwork(input_shape, action_size).to(device)
        self.qnetwork_target = QNetwork(input_shape, action_size)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE)
    
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        self.memory.add_exp(state, action, reward, next_state, done)
        
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
        states, actions, rewards, next_states, dones = experiences

        self.qnetwork_local.eval()
        Q_targets_next, _ = torch.max(self.qnetwork_local(next_states).detach(), dim = 1, keepdim = True)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        self.qnetwork_local.train()

        Q_expected = self.qnetwork_local(states).gather(1, actions)
        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()




