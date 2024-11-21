import numpy as np
import torch
from itertools import count

class ReplayBuffer1(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0
		self.state_dim = state_dim
		self.action_dim = action_dim

		self.state = np.zeros((max_size, self.state_dim))
		self.action = np.zeros((max_size, self.action_dim))
		self.next_state = np.zeros((max_size, self.state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)

	def cut_in(self, replay_buffer, batch_sample2):
		for i in range(batch_sample2):
			self.state[self.ptr] = replay_buffer.state[i]
			self.action[self.ptr] = replay_buffer.action[i]
			self.next_state[self.ptr] = replay_buffer.next_state[i]
			self.reward[self.ptr] = replay_buffer.reward[i]
			self.not_done[self.ptr] = replay_buffer.not_done[i]
			self.ptr = (self.ptr + 1) % self.max_size
			self.size = min(self.size + 1, self.max_size)
		replay_buffer.state = np.zeros((batch_sample2 , self.state_dim))
		replay_buffer.action = np.zeros((batch_sample2 , self.action_dim))
		replay_buffer.next_state = np.zeros((batch_sample2 , self.state_dim))
		replay_buffer.reward = np.zeros((batch_sample2, 1))
		replay_buffer.not_done = np.zeros((batch_sample2 ,1))
		replay_buffer.size = 0
		replay_buffer.ptr = 0
	def sample(self, batch_sample1):
		ind = np.random.randint(0, self.size, size=batch_sample1)
		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)
	def fill(self, env, initlen):
		while self.size < initlen:
			state = env.reset()
			for t in count():
				action = env.action_space.sample()
				next_state, reward, done, _ = env.step(action)
				done_bool = float(done) if t < env._max_episode_steps else 0
				self.state[self.ptr] = state
				self.action[self.ptr] = action
				self.next_state[self.ptr] = next_state
				self.reward[self.ptr] = reward
				self.not_done[self.ptr] = 1. - done_bool
				self.ptr = (self.ptr + 1) % self.max_size
				self.size = min(self.size + 1, self.max_size)
				state = next_state
				if done or t + 1 >= env._max_episode_steps:
					break

class ReplayBuffer2(object):
	def __init__(self, state_dim, action_dim, max_size=int(32)):
		self.max_size = max_size
		self.state_dim = state_dim
		self.action_dim	= action_dim
		self.ptr = 0
		self.size = 0
		self.state = np.zeros((max_size, self.state_dim))
		self.action = np.zeros((max_size, self.action_dim))
		self.next_state = np.zeros((max_size, self.state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)

	def sample(self, batch_size):
		ind  = np.random.choice(batch_size, batch_size, replace=False)
		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)