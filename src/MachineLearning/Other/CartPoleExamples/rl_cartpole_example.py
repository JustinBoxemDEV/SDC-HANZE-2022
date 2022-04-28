# Training a PPO model in cartpole using stable baselines 3
# https://stable-baselines3.readthedocs.io/en/master/

import os
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

environment_name = "CartPole-v0"
env = gym.make(environment_name)
env = DummyVecEnv([lambda: env])

# Load model
PPO_path = os.path.join("src/MachineLearning/Other/CartPoleExamples/cartpole_training", "training", "models", "PPO_Model_Cartpole")
model = PPO.load(PPO_path, env=env)

episodes = 5
for episode in range(episodes):
    obversation = env.reset()
    ep_reward = 0
    done = False
    
    while not done:
        env.render()
        action, _ = model.predict(obversation)
        next_obversation, reward, done, _ = env.step(action)
        
        ep_reward += reward
        obversation = next_obversation

    print("Episode: {}, Episode reward: {}".format(episode, ep_reward))
env.close()


# --------------------------------------------------------------------------------------------------
# In cmd run: tensorboard --logdir=. in the training/logs/PPO_1 folder to view the logs

# For training a PPO model in cartpole
# log_path = os.path.join("src/MachineLearning/Other/", "training", "logs")
# PPO_path = os.path.join("src/MachineLearning/Other/", "training", "models", "PPO_Model_Cartpole")

# environment_name = "CartPole-v0"
# env = gym.make(environment_name)
# env = DummyVecEnv([lambda: env])
# model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path)

# # Load model (comment line 46 if you use this)
# # model = PPO.load(PPO_path, env=env)

# # Train model
# model.learn(total_timesteps=200000)
# model.save(PPO_path)

# print(evaluate_policy(model, env, n_eval_episodes=10, render=True))