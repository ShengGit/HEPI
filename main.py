import numpy as np
import pandas as pd
import torch
import gym
import argparse
import os
import matplotlib.pyplot as plt
import utils
import TD3
import HEPI
import OurDDPG
import DDPG
from torch.utils.tensorboard import SummaryWriter
avg_reward = 0
reward_list = []
np.set_printoptions(threshold=np.inf)
def eval_policy(policy, env_name, seed, eval_episodes=10):
	eval_env = gym.make(env_name)
	eval_env.seed(seed + 100)
	reward_list = []
	min_reward = 0.
	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		rewards = 0.
		while not done:
			action = policy.select_action(np.array(state))
			state, reward, done, _ = eval_env.step(action)
			rewards += reward
		reward_list.append(rewards)
	avg_reward = sum(reward_list) / eval_episodes
	max_reward = max(reward_list)


	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f} max：{max_reward:.3f}")
	print("---------------------------------------")
	return avg_reward, max_reward, reward_list

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--policy", default="HEPI")
	parser.add_argument("--env", default="Ant-v2")
	parser.add_argument("--seed", default=1, type=int)
	parser.add_argument("--start_timesteps", default=35e3, type=int)
	parser.add_argument("--eval_freq", default=5e3, type=int)
	parser.add_argument("--max_timesteps", default=1e6, type=int)
	parser.add_argument("--expl_noise", default=0.1)
	parser.add_argument("--batch_size", default=256, type=int)
	parser.add_argument("--batch_sample1", default=224, type=int)
	parser.add_argument("--batch_sample2", default=32, type=int)
	parser.add_argument("--discount", default=0.99)
	parser.add_argument("--tau", default=0.005)
	parser.add_argument("--policy_noise", default=0.2)
	parser.add_argument("--noise_clip", default=0.5)
	parser.add_argument("--policy_freq", default=2, type=int)
	parser.add_argument("--save_model", action="store_true")
	parser.add_argument("--load_model", default="")
	args = parser.parse_args()

	file_name = f"{args.policy}_{args.env}_{args.seed}"
	eval_name = f"{args.policy}_{args.env}_{args.seed}"
	print("---------------------------------------")
	print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
	print("---------------------------------------")

	if not os.path.exists("./results"):
		os.makedirs("./results")

	if not os.path.exists("./models"):
		os.makedirs("./models")
	if not os.path.exists("./results_max"):
		os.makedirs("./results_max")
	if not os.path.exists("./learning_curves"):
		os.makedirs("./learning_curves")
	env = gym.make(args.env)

	env.seed(args.seed)
	env.action_space.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)


	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0]
	max_action = float(env.action_space.high[0])

	kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"discount": args.discount,
		"tau": args.tau,
	}

	if args.policy == "HEPI":
		kwargs["policy_noise"] = args.policy_noise * max_action
		kwargs["noise_clip"] = args.noise_clip * max_action
		kwargs["policy_freq"] = args.policy_freq
		policy = HEPI.HEPI(**kwargs)

	if args.load_model != "":
		policy_file = file_name if args.load_model == "default" else args.load_model
		policy.load(f"./models/{policy_file}")

	replay_buffer1 = utils.ReplayBuffer1(state_dim, action_dim)
	replay_buffer2 = utils.ReplayBuffer2(state_dim, action_dim)

	eval_return = eval_policy(policy, args.env, args.seed)
	evaluations_avg = [eval_return[0]]
	evaluations_max = [eval_return[1]]
	evaluations_list = [eval_return[2]]
	batch_sample1 = args.batch_sample1
	batch_sample2 = args.batch_sample2
	state, done = env.reset(), False
	episode_reward = 0
	episode_timesteps = 0
	episode_num = 0
	res = []

	for t in range(int(args.max_timesteps)):
		episode_timesteps += 1
		if t < args.start_timesteps:
			action = env.action_space.sample()
		else:
			action = (
				policy.select_action(np.array(state))
				+ np.random.normal(0, max_action * args.expl_noise, size=action_dim)
			).clip(-max_action, max_action)

		next_state, reward, done, _ = env.step(action)
		done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

		if t < args.start_timesteps and reward < avg_reward:
			replay_buffer1.add(state, action, next_state, reward, done_bool)
		else:
			reward_list.append(reward)
			replay_buffer2.add(state, action, next_state, reward, done_bool)

		state = next_state
		episode_reward += reward

		if (t >= args.start_timesteps and replay_buffer2.size >= batch_sample2) or done_bool == done:
			for i in range(batch_sample2):
				policy.train(replay_buffer1, replay_buffer2, args.batch_size-replay_buffer2.size, replay_buffer2.size)
			avg_reward = np.mean(reward_list)
			replay_buffer1.cut_in(replay_buffer2, batch_sample2)
		if done:
			print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
			res.append(episode_reward)
			state, done = env.reset(), False
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1
		if (t + 1) % args.eval_freq == 0:
			ret = eval_policy(policy, args.env, args.seed)
			evaluations_avg.append(ret[0])
			evaluations_max.append(ret[1])
			evaluations_list.append(ret[2])
			np.save(f"./results/16{file_name}", evaluations_avg)
			np.save(f"./results_max/16max{file_name}", evaluations_max)
			np.save(f"./learning_curves/16{eval_name}", evaluations_list)
			if ret[1] >= max(evaluations_max):
				policy.save(f"./models/16{file_name}")
	env.close()

print("done!")