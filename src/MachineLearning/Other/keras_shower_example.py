# Keras example (Not functional as a whole)
from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random

# environment
class ShowerEnv(Env):
    def __init__(self):
        self.action_space = Discrete(3) # 0 = down, 1 = steady, 2 = up
        self.observation_space = Box(low=np.array([0]), high=np.array([100]))
        self.state = 38 + random.randint(-3, 3)
        self.shower_length = 60

    def step(self, action):
        self.state += action -1
        self.shower_length -= 1

        if self.state >= 37 and self.state <=39:
            reward =1
        else:
            reward = -1

        if self.shower_length <= 0:
            done = True
        else:
            done = False

        # Random noise
        self.state += random.randint(-1,1)
        info = {}
        
        return self.state, reward, done, info
        
    def render(self):
        pass
    
    def reset (self):
        self.state = 38 + random.randint(-3,3)
        self.shower_length = 60
        return self.state


env = ShowerEnv()
episodes = 10
for episode in range(10):
    state = env.reset()
    done = False
    score = 0

    while not done:
        action = env.action_space.sample()
        n_state, reward, done, info = env.step(action)
        score+= reward
    print("Episode: {} Score: {}".format(episode, score))


    # model
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.optimizers import Adam

    states = env.observation_space.shape
    actions = env.action_space.n

    def build_model(states, actions):
        model = Sequential()
        model.add(Dense(24, activation="relu", input_shape=states)) # pass the temperature
        model.add(Dense(24, activation="relu"))
        model.add(Dense(actions, activation="linear"))
        return model

model = build_model(states, actions)
model.summary()
