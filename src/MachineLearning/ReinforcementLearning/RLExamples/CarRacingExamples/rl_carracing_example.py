""" Reinforcement Learning example: training a PPO model in carracing using stable baselines 3
    
    In cmd run: tensorboard --logdir=. in the carracing_training/logs/PPO_1 folder to view the logs

    Needs swig, python 3.7 and gym 0.21:
        conda install -c conda swig
        conda create -n p37 python=3.7
        pip install gym[box2d]=0.21
        
    Sources:
    https://stable-baselines3.readthedocs.io/en/master/
    https://github.com/openai/gym
"""
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import os


# TEST MODEL
environment_name = "CarRacing-v0"
env = gym.make(environment_name)
env = DummyVecEnv([lambda: env]) # for making several instances

# Load model
log_path = os.path.join("src/MachineLearning/Other/CarRacingExamples/carracing_training/", "training", "logs")
ppo_path = os.path.join("src/MachineLearning/Other/CarRacingExamples/carracing_training/", "models", "PPO_Driving_Model")
model = PPO.load(ppo_path, env=env, device='cuda')
print(env.action_space)

episodes = 5
for episode in range(episodes):
    observation = env.reset()
    done = False
    score = 0

    while not done:
        env.render()
        action = model.predict(observation)
        # print(f"action: {action}")
        observation, reward, done, _ = env.step(action[0])
        score += reward
    print("Episode: {} Score: {}".format(episode, score))
env.close()

# --------------------------------------------------------------------------------------------------
# TRAIN MODEL

# log_path = os.path.join("src/MachineLearning/Other/", "training", "logs")
# ppo_path = os.path.join("src/MachineLearning/Other/", "training", "models", "PPO_Driving_Model")

# environment_name = "CarRacing-v0"
# env = gym.make(environment_name)
# env = DummyVecEnv([lambda: env])
# env.render()

# model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=log_path, device="cuda")

# # Load model (comment line 50 if you use this)
# # model = PPO.load(ppo_path, env=env)

# # Train model
# model.learn(total_timesteps=50000)
# model.save(ppo_path)


# print(evaluate_policy(model, env, n_eval_episodes=10, render=True))