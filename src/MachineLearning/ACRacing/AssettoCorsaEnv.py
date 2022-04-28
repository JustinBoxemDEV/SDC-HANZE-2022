# Steal from gym Carla implementation https://github.com/cjy1992/gym-carla/blob/master/gym_carla/envs/carla_env.py
# Brainstorm version

import gym
from gym import spaces
import numpy as np

class AssettoCorsaEnv(gym.Env):
    def __init__(self):
        print("Assetto Corsa Environment")

        self.action_space = spaces.Box(
            np.array([-1, 0, 0]).astype(np.float32), # steer
            np.array([+1, +1, +1]).astype(np.float32), # gas
            # could add brake here
        )

        self.observation_space = None # not sure yet
        self.reward = 0
        
    def reset(self):
        # TODO: Use reset script (AC reset shortkey)

        obversation = None # Screen capture
        return obversation

    def step(self, action):
        # TODO: calculate acceleration and steering (ignore braking)

        # TODO: replace Temp values
        observation = None
        reward = self.get_reward()
        done = False
        info = {}
        return observation, reward, done, info

    def render(self):
        # Don't need this i think?
        pass

    def get_reward(self):
        # Negative points for driving too slow
        # Negative points for driving on grass (green pixels)
        # Positive points for driving on the track (grey pixels)
        # Positive points for finishing track faster than previous time
        reward = 0
        return reward

    def detect_collision(self):
            pass