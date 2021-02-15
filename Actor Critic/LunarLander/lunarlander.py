import gym
from model import Agent
import torch
import matplotlib.pyplot as plt
import time

def train(game, num_episodes, agent):
    torch.manual_seed(543)
    for k in range(num_episodes):
        done = False
        state = game.reset()
        score = 0
        while not done:
            action = agent.get_action(state, sample = True)
            next_state, reward, done, _ = game.step(action)
            agent.rewards.append(reward)
            state = next_state
            score += reward
        agent.learn()
        agent.clearmemory()
        print(f'Episode {k} Score {score}')
        if score > 200:
            torch.save(agent.network.state_dict(), "./models/LunarLander.pth")

def infer(game, agent, random = False, path = './models/LunarLander.pth'):
    agent.clearmemory()
    if not random:
        agent.network.load_state_dict(torch.load(path))
    for k in range(5):
        done = False
        state = game.reset()
        score = 0
        while not done:
            game.render()
            action = agent.get_action(state, sample = False)
            next_state, reward, done, _ = game.step(action)
            agent.rewards.append(reward)
            state = next_state
            score += reward
            time.sleep(0.015)
        print(f'Episode {k} Score {score}')
        





if __name__=='__main__':
    #pip install gym[Box2D] -- maybe needed
    game = gym.make('LunarLander-v2')
    action_size = game.action_space.n
    print("Action size ", action_size)

    state_size = game.observation_space.shape[0]
    print("State size ", state_size)
    num_episodes = 2000
    agent = Agent(state_size, action_size, gamma = 0.99, fc1 = 64, fc2 = 64)

    #train(game, num_episodes, agent)
    infer(game, agent, random = False)
    