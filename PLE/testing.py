# Import Flappy Bird from games library in
# Python Learning Environment (PLE)
from ple.games.flappybird import FlappyBird 
 
# Import PyGame Learning Environment
from ple import PLE 
 
# Import Numpy
import numpy as np
 
class NaiveAgent():
  """
  This is a naive agent that just picks actions at random.
  """
  def __init__(self,actions):
    self.actions = actions
 
  def pickAction(self, reward, obs):
    return self.actions[np.random.randint(0, len(self.actions))]
 
# Create a game instance
game = FlappyBird() 
 
# Pass the game instance to the PLE
p = PLE(game)
 
# Create the agent
agent = NaiveAgent(p.getActionSet())
 
# Initialize the environment
p.init()
 
actions = p.getActionSet()
action_dict = {0: actions[1], 1:actions[0]}
 
reward = 0.0
 
for f in range(15000):
 
  # If the game is over
  if p.game_over():
    p.reset_game()
 
  action = agent.pickAction(reward, p.getScreenRGB())
  reward = p.act(action)
 
  if f > 1000:
    p.display_screen = True
    p.force_fps = False # Slow screen