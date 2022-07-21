""" Reinforcement Learning example: random carracing actions

    needs swig, python 3.7 and gym 0.21:
    conda install -c conda swig
    conda create -n p37 python=3.7
    pip install gym[box2d]=0.21
    
    Sources:
    https://github.com/openai/gym
"""
import gym

environment_name = "CarRacing-v0"
env = gym.make(environment_name)

episodes = 5
for episode in range(episodes):
    observation = env.reset()
    done = False
    score = 0

    while not done:
        env.render()
        action = env.action_space.sample() # random actions!
        observation, reward, done, _ = env.step(action)
        score += reward
    print("Episode: {} Score: {}".format(episode, score))
env.close()