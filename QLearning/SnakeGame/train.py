from SnakeGame import SnakeGame
import random
import numpy as np
import cv2

game = SnakeGame((20,20))
states = 2**8 - 2**4
actions = 4

margin = 15     #=decimal([0,0,0,0,1,1,1,1])
best_q = None
'''
states vary from [0,0,0,0,0,0,0,0] to [1,1,1,1,1,1,1,1] - notice binary nos
so in the table the index of each state will be its decimal value = summation(2**i * state[i])
'''
def state_index(state):
    val = 0
    for i in range(8):
        val += (2**i)*state[7-i]
    return int(val-margin)

def test(qtable, render = False):
    state = game.reset_board()
    state = state_index(state)
    done = False
    while not done:
        if render:
            game.render()
        action = np.argmax(qtable[state,:])
        next_state, reward, done = game.step(action)
        next_state = state_index(next_state)
        state = next_state
    return game.pellets

def train():
    max_pellets = 0
    num_episodes = 3000
    gamma = 0.9
    eps = 0.2
    lr = 0.2
    qtable = np.zeros((states, actions))

    for i in range(num_episodes):
        state = game.reset_board()
        state = state_index(state)
        done = False
        while not done:
            #uncomment if you want to see training, warning:it will slow down training
            #game.render()
            if random.random() > eps:
                action = np.argmax(qtable[state,:])
            else:
                action = random.choice(range(actions))
            next_state, reward, done = game.step(action)
            next_state = state_index(next_state)
            
            tderror = reward + (1-done)*gamma*np.max(qtable[next_state,:]) - qtable[state, action]
            qtable[state,action] = qtable[state,action] + lr*tderror
            state = next_state
        if i%100 == 0:
            pellets= test(qtable)
            if pellets > max_pellets:
                print(f"Score : {pellets}")
                max_pellets = pellets
                best_q = qtable
    return qtable


if __name__ == '__main__':
    qtable = train()
    test(qtable, True)



