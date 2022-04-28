# if you get the "Module time has no attribute clock" error, install python 3.8.0 (the module time was removed after this python verison)

import gym
env = gym.make('CartPole-v0')
state = env.reset()

# print("State: {}, Reward: {}, Done: {}, Info: {}, Next state: {}".format(state, reward, done, info, next_state))
# print("Action space: {}".format(env.action_space))
# print("Observation space: {}".format(env.observation_space))

episodes = 100
for ep_cnt in range(episodes):
    step_cnt = 0
    ep_reward = 0
    done = False
    state = env.reset()

    while not done:
        next_state, reward, done, _ = env.step(env.action_space.sample()) # random actions
        env.render()
        step_cnt += 1
        ep_reward += reward
        state = next_state

    print("Episode: {}, Step count: {}, Episode reward: {}".format(ep_cnt, step_cnt, ep_reward))
env.close()
