from ple.games.flappybird import FlappyBird 

from ple import PLE 
 
import numpy as np

import matplotlib.pyplot as plt
import time
import cv2
from model import Agent
import torch


input_channels = 4
input_shape = (80,80)

def preprocess(game_state, size = input_shape):
  game_state = np.transpose(game_state, (1,0))
  _, game_state = cv2.threshold(game_state, 142, 255, cv2.THRESH_BINARY)
  game_state = cv2.resize(game_state, size)
  game_state = np.expand_dims(game_state, axis = 0)
  game_state = game_state/255
  
  return game_state
 
game = FlappyBird() 
 
flappy = PLE(game, display_screen=False, force_fps = True)
 
flappy.init()
 
actions_list = flappy.getActionSet()

max_rew = 0
eps = 1
eps_decay = 0.9995
eps_min = 0.01
agent = Agent(action_size = 2, input_shape = input_shape, input_channels = input_channels)
scores = []
for i in range(15000):

  done = False
  flappy.reset_game()
  screen = flappy.getScreenGrayscale()
  screen = preprocess(screen)
  state = np.concatenate([screen]*input_channels, axis = 0)
  score = 0
  pipes = 0
  while not done:
    action_idx = agent.get_action(state, eps)
    action = actions_list[action_idx]
    reward = flappy.act(action)

    if reward == 0:
      reward = 1
    elif reward == 1:
      reward = 10
      pipes+=1
    else:
      reward = -100

    screen = flappy.getScreenGrayscale()
    next_screen = preprocess(screen)
    
    next_state = np.append(state[1:], next_screen, axis = 0)
    if flappy.game_over():
      done = True

    agent.step(state, action_idx, reward, next_state, done)
    state = next_state
    score+=reward
    eps = max(eps_min, eps*eps_decay)
  scores+=[pipes]
  print(f'Episode {i} Score {pipes}')
  if pipes>=max_rew:
      torch.save(agent.qnetwork_local.state_dict(), f'./flappy{pipes}.pth')
      max_rew = pipes

plt.scatter(y = scores, x = range(len(scores)))
plt.savefig('./img.png')





  
