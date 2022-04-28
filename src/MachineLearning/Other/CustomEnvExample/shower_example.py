# Keras example (Not functional as a whole)
from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random
import os
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

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
        # For visualization
        pass
    
    def reset (self):
        self.state = 38 + random.randint(-3,3)
        self.shower_length = 60
        return self.state

# Test with gym + random
# env = ShowerEnv()
# episodes = 5
# for episode in range(episodes):
#     obversation = env.reset()
#     done = False
#     score = 0

#     while not done:
#         env.render()
#         action = env.action_space.sample()
#         obversation, reward, done, info = env.step(action)
#         score+= reward
#     print("Episode: {} Score: {}".format(episode, score))
# env.close()

# --------------------------------------------------------------------------------------------------------------------------------------------

# Train with gym + stable baselines PPO model
# ppo_path = os.path.join("src/MachineLearning/Other/CustomEnvExample/", "shower_training", "models", "PPO_Shower_Model")
# log_path = os.path.join("src/MachineLearning/Other/CustomEnvExample/", "shower_training", "logs")
# env = ShowerEnv()
# model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_path)
# model.learn(total_timesteps=50000)
# model.save(ppo_path)

# Test with gym + stable baselines PPO model
ppo_path = os.path.join("src/MachineLearning/Other/CustomEnvExample/", "shower_training", "models", "PPO_Shower_Model")
env = ShowerEnv()
model = PPO.load(ppo_path, env)
print("Mean episode reward and standard deviation:", evaluate_policy(model, env, n_eval_episodes=10, render=False))

# --------------------------------------------------------------------------------------------------------------------------------------------

# With gym + keras model (has some bug idk)
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.optimizers import Adam
# from rl.agents.dqn import DQNAgent
# from rl.policy import BoltzmannQPolicy
# from rl.memory import SequentialMemory

# env = ShowerEnv()

# states = env.observation_space.shape
# print(f"states: {states}")
# actions = env.action_space.n

# def build_model(states, actions):
#     model = Sequential()
#     model.add(Dense(24, activation="relu", input_shape=states)) # pass the temperature
#     model.add(Dense(24, activation="relu"))
#     model.add(Dense(actions, activation="linear"))
#     return model

# model = build_model(states, actions)
# # model.summary()

# def build_agent(model, actions):
#     policy = BoltzmannQPolicy()
#     memory = SequentialMemory(limit=5000, window_length=1)
#     dqn = DQNAgent(model=model, memory=memory, policy=policy, nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2)
#     return dqn

# dqn = build_agent(model, actions)
# dqn.compile(Adam(lr=1e-3), metrics=['mse'])
# dqn.fit(env, nb_steps=5000, visualize=False, verbose=1)

# scores = dqn.test(env, nb_episodes=100, visualize=False)
# print(np.mean(scores.history['episode_reward']))
