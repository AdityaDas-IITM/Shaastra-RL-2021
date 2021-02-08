import numpy as np
import random
import gym
'''
    Passenger locations:
    - 0: R(ed)
    - 1: G(reen)
    - 2: Y(ellow)
    - 3: B(lue)
    - 4: in taxi
    Destinations:
    - 0: R(ed)
    - 1: G(reen)
    - 2: Y(ellow)
    - 3: B(lue)
    Actions:
    There are 6 discrete deterministic actions:
    - 0: move south
    - 1: move north
    - 2: move east
    - 3: move west
    - 4: pickup passenger
    - 5: drop off passenger
    Rewards:
    There is a default per-step reward of -1,
    except for delivering the passenger, which is +20,
    or executing "pickup" and "drop-off" actions illegally, which is -10.
    Rendering:
    - blue: passenger
    - magenta: destination
    - yellow: empty taxi
    - green: full taxi
    - other letters (R, G, Y and B): locations for passengers and destinations
    state space is represented by:
        (taxi_row, taxi_col, passenger_location, destination)
'''

action_size = env.action_space.n
print("Action size ", action_size)

state_size = env.observation_space.n
print("State size ", state_size)

env = gym.make("Taxi-v2")
env.render()

qtable = np.zeros((state_size, action_size))

total_episodes = 50000        # Total episodes
max_steps = 99                # Max steps per episode

lr = 0.7                      # Learning rate
gamma = 0.618                 # Discounting rate

eps = 1.0                     # Exploration rate
min_eps = 0.01                # Minimum exploration probability
eps_decay = 0.99              # Exponential decay rate for exploration prob
eval_every = 100              # Evaluation freq

bestq = None                  # Store best qtable

def train():
    max_rew = 0
    step = 0
    for episode in range(total_episodes):
        state = env.reset()
        done = False

        for step in range(max_steps):
            if random.random() > eps:
                action = np.argmax(qtable[state,:])

            else:
                action = env.action_space.sample()

            next_state, reward, done, _= env.step(action)

            tderror = reward + gamma*(1-done)*np.max(qtable[next_state,:]) - qtable[state, action]
            qtable[state,action] = qtable[state,action] + lr*tderror
            state = next_state

            if step%eval_every == 0:
                reward = infer(5, qtable)
                if reward>=max_rew:
                    max_rew = reward
                    bestq = qtable    
            step+=1        
            if done == True:
                break
            eps = max(min_eps, eps*eps_decay)
    return bestq

def infer(episodes, qtable, render = False):
    env.reset()
    rewards = 0

    for episode in range(episodes):
        state = env.reset()
        done = False

        for step in range(max_steps):
            if render:
                env.render()
            action = np.argmax(qtable[state,:])

            new_state, reward, done, _ = env.step(action)

            rewards += reward

            if done:
                break
            state = new_state
    env.close()
    avg_rew = sum(reward)/episodes
    return avg_rew

final_q = train()
infer(episodes = 5, qtable = final_q, render = True)