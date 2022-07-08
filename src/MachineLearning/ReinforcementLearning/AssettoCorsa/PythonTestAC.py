""" Testing loop for the neural network in Assetto Corsa (Virtual Enviroment)

    This is a gym + stabebaselines3 implementation inspired by a gym Carla implementation.
    This approach was experimental and did not end up getting used.

    Sources:
    https://github.com/openai/gym
    https://stable-baselines3.readthedocs.io/en/master/
    https://github.com/cjy1992/gym-carla/blob/master/gym_carla/envs/carla_env.py
"""

# Test the model
import torch
import os
from AssettoCorsaEnv import AssettoCorsaEnv
from stable_baselines3 import PPO

save_path = os.path.join("src/MachineLearning/AssettoCorsa/", "training_data", "models", "best_model_080522")

torch.cuda.empty_cache()
env = AssettoCorsaEnv()
model = PPO.load(save_path, env=env, device='cuda')

print('testing...')
episodes = 5
for episode in range(episodes):
    observation = env.reset()
    done = False
    score = 0

    while not done:
        # env.render() # does nothing for AC
        action = model.predict(observation)
        # print(f"action: {action}")
        observation, reward, done, _ = env.step(action[0])
        score += reward
    print("Episode: {} Score: {}".format(episode, score))
env.close()

# ------------------------------------------------------------------------------------
# Random values test
# from AssettoCorsaEnv import AssettoCorsaEnv

# env = AssettoCorsaEnv()
# # env.reset()
# # env.step()

# episodes = 5
# for episode in range(episodes):
#     observation = env.reset()
#     done = False
#     score = 0

#     while not done:
#         action = env.action_space.sample() # random actions
#         observation, reward, done, _ = env.step(action)
#         score += reward
#     print("Episode: {} Score: {}".format(episode, score))

