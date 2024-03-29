""" Training loop for the neural network in Assetto Corsa (Virtual Enviroment)

    This is a gym + stabebaselines3 implementation inspired by a gym Carla implementation.
    This approach was experimental and did not end up getting used.

    Sources:
    https://github.com/openai/gym
    https://stable-baselines3.readthedocs.io/en/master/
    https://github.com/cjy1992/gym-carla/blob/master/gym_carla/envs/carla_env.py

"""

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
import os
from AssettoCorsaEnv import AssettoCorsaEnv

# Maybe if i register the game in Gym this is needed
# environment_name = "AssettoCorsa-v0"
# env = gym.make(environment_name)

# might be redundant, not sure yet
env = AssettoCorsaEnv()
env = DummyVecEnv([lambda: env])

# Paths
log_path = os.path.join("src/MachineLearning/ACRacing/", "training_data", "logs")
save_path = os.path.join("src/MachineLearning/ACRacing/", "training_data", "models", "AC_model_100522") # location for best model

# Load model (to resume training)
# model = PPO.load(save_path, env=env)

# Create new model for training (maybe try MultiInputPolicy)
model = PPO("CnnPolicy", env, verbose=1, 
            tensorboard_log=log_path, device="cuda", # change device to cpu if you dont have a gpu
            n_epochs=10, n_steps=1024, batch_size=8, use_sde=True)  # TODO: change these values

# With callback
stop_callback = StopTrainingOnRewardThreshold(reward_threshold=700, verbose=1)
eval_callback = EvalCallback(env, callback_on_new_best=stop_callback, eval_freq=2, best_model_save_path=save_path, verbose=1)

# Train model (evaluate every x)
model.learn(total_timesteps=2, callback=eval_callback, eval_env=env)

# without callback (for training short periods)
# torch.cuda.empty_cache()
# model = model.learn(total_timesteps=1000)
# model.save(save_path)

print("we made it")
env.close()