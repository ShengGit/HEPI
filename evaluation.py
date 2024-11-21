import HEPI
import TD3
import DDPG
import numpy as np
import pandas as pd
import torch
import gym
def eval_policy(filename, env, seed, eval_epsiodes):
    step = 0
    eval_env = gym.make(env)
    eval_env = gym.wrappers.Monitor(eval_env, './test', video_callable=lambda episode_id: True, force=True)
    eval_env.seed(seed + 10)
    res_list = []
    state, done = eval_env.reset(), False
    for i in range(1000):
        rewards = 0.
        while i<1000:
            step += 1
            action = eval_env.action_space.sample()
            state, reward, done, _ = eval_env.step(action)
            rewards += reward
        res_list.append(rewards)
        print(step)
    avg_reward = sum(res_list) / 200
    max_reward = max(res_list)
    print(res_list)

    print('---------------------------------')
    print(f'Evaluation over {eval_epsiodes} average {avg_reward:3f} max {max_reward:3f}')
    print('---------------------------------')
eval_policy('./comparison/DDPG_Hopper-v2', 'Hopper-v2', 42, 5)


