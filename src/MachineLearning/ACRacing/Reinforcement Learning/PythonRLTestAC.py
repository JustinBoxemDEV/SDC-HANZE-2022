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

# ------------------------------------------------------------------------------------
# Model test
import torch
import os
from AssettoCorsaEnv import AssettoCorsaEnv
from stable_baselines3 import PPO

save_path = os.path.join("src/MachineLearning/ACRacing/", "training_data", "models", "best_model_080522")

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
        # env.render()
        action = model.predict(observation)
        # print(f"action: {action}")
        observation, reward, done, _ = env.step(action[0])
        score += reward
    print("Episode: {} Score: {}".format(episode, score))
env.close()