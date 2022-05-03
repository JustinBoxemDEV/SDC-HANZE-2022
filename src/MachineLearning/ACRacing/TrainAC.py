# Training loop for the nn in Assetto Corsa (Virtual environment)
# Brainstorm version

import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
import torch

import os
from AssettoCorsaEnv import AssettoCorsaEnv

# Maybe if i register the game in Gym this is needed
# environment_name = "AssettoCorsa-v0"
# env = gym.make(environment_name)

# might be redundant, not sure yet
env = AssettoCorsaEnv()
env = DummyVecEnv([lambda: env])

# Paths
log_path = os.path.join("src/MachineLearning/ACRacing/", "training", "logs")
save_path = os.path.join("src/MachineLearning/ACRacing/", "training", "models", "AC_model") # location for best model

# Load model (to resume training)
# model = PPO.load(save_path, env=env)

# Create new model for training (maybe try MultiInputPolicy)
model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=log_path, device="cuda", n_steps=16, batch_size=8) # change device to cpu if you dont have a gpu

# TODO: set callback
# stop_callback = StopTrainingOnRewardThreshold(reward_threshold=1000, verbose=1)
# eval_callback = EvalCallback(env, callback_on_new_best=stop_callback, eval_freq=2, best_model_save_path=save_path, verbose=1)

# Train model (evaluate every x)
# model.learn(total_timesteps=2, callback=eval_callback, eval_env=env)

torch.cuda.empty_cache()
model = model.learn(total_timesteps=100)
model.save(save_path)
print("we did it")
env.close()


# torch.cuda.empty_cache()
# print('testing...')
# episodes = 5
# for episode in range(episodes):
#     observation = env.reset()
#     done = False
#     score = 0

#     while not done:
#         # env.render()
#         action = model.predict(observation)
#         # print(f"action: {action}")
#         observation, reward, done, _ = env.step(action[0])
#         score += reward
#     print("Episode: {} Score: {}".format(episode, score))
# env.close()

