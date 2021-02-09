import gym
import random
import time
import cv2
import numpy as np
from model import Agent
import torch



def train(game, episodes, eps, eps_decay, agent, resume = False, path = ''):
    if resume:
        agent.qnetwork_local.load_state_dict(torch.load(path))
    for i in range(episodes):
        state = game.reset()
        total_rew = 0
        for _ in range(5000):
            
            action = agent.get_action(state, eps)
            next_state, reward, done, _ = game.step(action)
            reward = reward if not done else -reward
            agent.step(state, action, reward, next_state, done)
            state = next_state
            eps*=eps_decay
            total_rew+=reward
            if done:
                break
        print (f'Episode {i} Average Reward {total_rew}')

        if total_rew >= 195:
            torch.save(agent.qnetwork_local.state_dict(), f'./checkpoints/cartpole.pth')


def infer(game, agent, episodes, arbit = False, path = ''):
    if not arbit:
        agent.qnetwork_local.load_state_dict(torch.load(path))
    for i in range(episodes):
        rew = 0
        state = game.reset()
        for _ in range(1000):
            game.render()
            time.sleep(0.02)
            if not arbit:
                action = agent.get_action(state, eps=0)
            else:
                action = random.choice(range(action_size))
            state, reward, done, _ = game.step(action)
            rew+=reward
            if done:
                break
            

if __name__ == '__main__':
    game = gym.make('CartPole-v0')
    action_size = game.action_space.n
    print("Action size ", action_size)

    state_size = game.observation_space.shape[0]
    print("State size ", state_size)
    eps = 1
    eps_decay = 0.99
    min_epc = 0.1
    agent = Agent(state_size, action_size)
    max_rew = -1e5
    episodes = 5000

    #train(game, episodes, eps, eps_decay, agent)
    infer(game, agent, episodes = 5, arbit = False, path = './checkpoints/cartpole.pth')
