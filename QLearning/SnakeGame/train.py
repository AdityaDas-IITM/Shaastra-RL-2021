from SnakeGame import SnakeGame
import random
import numpy as np
import cv2

game = SnakeGame((20,20))
states = 2**8
actions = 4

best_q = None
'''
states vary from [0,0,0,0,0,0,0,0] to [1,1,1,1,1,1,1,1] - notice binary nos
so in the table the index of each state will be its decimal value = summation(2**i * state[i])
'''
def state_index(state):
    val = 0
    for i in range(8):
        val += (2**i)*state[i]
    return int(val)

def test(qtable, render = False):
    state = game.reset_board(save_frames = True)
    state = state_index(state)
    done = False
    for i in range(10000):
        if render:
            game.render()
        action = np.argmax(qtable[state,:])
        next_state, reward, done = game.step(action)
        next_state = state_index(next_state)
        state = next_state
        if done:
            break
    game.create_vid('./snake.mp4')
    return game.pellets

def train():
    max_pellets = 0
    num_episodes = 3000
    gamma = 0.9
    eps = 0.2
    lr = 0.2
    qtable = np.zeros((states, actions))

    for i in range(num_episodes):
        state = game.reset_board(save_frames = False)
        state = state_index(state)
        done = False
        for _ in range(10000):
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
            if done:
                break
        if i%100 == 0:
            pellets= test(qtable)
            if pellets > max_pellets:
                print(f"Score : {pellets}")
                max_pellets = pellets
                best_q = qtable
    return qtable


if __name__ == '__main__':
    qtable = train()
    print("Training done!!")
    test(qtable, False)



