
# TEST FILE
from AssettoCorsaEnv import AssettoCorsaEnv

env = AssettoCorsaEnv()
# env.reset()
# env.step()

episodes = 5
for episode in range(episodes):
    observation = env.reset()
    done = False
    score = 0

    while not done:
        action = env.action_space.sample() # random actions
        observation, reward, done, _ = env.step(action)
        score += reward
    print("Episode: {} Score: {}".format(episode, score))
    