# -*- coding: utf-8 -*-
# @Software: PyCharm 
# @Time    : 2023/12/25 16:43
# @Author  : Jane_Sang
from env import TaskOffloadingEnv
import datetime
import gym
import random
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('./log/{}'.format(datetime.datetime.now().__str__().replace(':', '_'))+" Random")

# n_ue = 5
n_bs = 2
n_en = 10
# n_task = 5
max_episodes = 10000
episode_rewards = []
total_reward = 0
n_offloading = n_en + n_bs + 1
max_steps = 1000
env = TaskOffloadingEnv(use_en=True, use_bs=True)
for episode in range(max_episodes):
    episode_reward = 0
    state = env.reset()
    for step in range(max_steps):
        action = random.randint(0, n_offloading-1)
        next_state, reward, done, _ = env.step(action)
        episode_reward += reward
        if done or step == max_steps - 1:
            episode_rewards.append(episode_reward)
            print("Episode " + str(episode) + ": " + str(episode_reward))
            break
        state = next_state
    total_reward += episode_reward
    writer.add_scalar('episode/reward', episode_reward, episode + 1)
    writer.add_scalar('episode/avg-reward', total_reward / (episode + 1), episode + 1)