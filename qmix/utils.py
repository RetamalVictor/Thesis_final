import collections
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



class ReplayBuffer(object):
    def __init__(self, buffer_limit, chunk_size, n_agents, input_shape, batch_size=32):
        
        self.buffer = collections.deque(maxlen=buffer_limit)

        self.state_memory = torch.zeros((batch_size, chunk_size, n_agents, *input_shape), dtype=torch.float32).cuda()
        self.new_state_memory = torch.zeros((batch_size, chunk_size, n_agents, *input_shape), dtype=torch.float32).cuda()
        self.action_memory = torch.zeros((batch_size, chunk_size, n_agents), dtype=torch.float32).cuda()
        self.reward_memory = torch.zeros((batch_size,chunk_size, n_agents), dtype=torch.float32).cuda()
        self.terminal_memory = torch.zeros((batch_size, chunk_size, 1), dtype=torch.float32).cuda()

    def put(self, transition):
        """
         this transition a list of tensors.
         transition = [s, a, r, s_prime, done]
            s = [n_agents, obs_size]: obs_size = (NearestNeigbors)
            a = [n_agents]
            r = [n_agents]
            s_prime = [n_agents, obs_size]
            done = [1]
        """
        self.buffer.append(transition)


    def sample_chunk(self, batch_size, chunk_size):
        start_idx = np.random.randint(0, len(self.buffer) - chunk_size, batch_size)
        # s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [], [], []

        batch = 0
        for idx in start_idx:
            count = 0
            for chunk_step in range(idx, idx + chunk_size):
                s, a, r, s_prime, done = self.buffer[chunk_step]
                self.state_memory[batch, count] = s
                self.action_memory[batch, count] = a
                self.reward_memory[batch, count] = r[:,0]
                self.new_state_memory[batch, count] = s_prime
                self.terminal_memory[batch, count] = done[0]
                count += 1

                # s_lst.append(s)
                # a_lst.append(a)
                # r_lst.append(r)
                # s_prime_lst.append(s_prime)
                # done_lst.append(done)
            batch += 1

        # n_agents, obs_size = len(s_lst[0]), len(s_lst[0][0])
        # return a clone of the tensor
        return self.state_memory.clone(), \
                self.action_memory.clone(), \
                self.reward_memory.clone(), \
                self.new_state_memory.clone(), \
                self.terminal_memory.clone()

        # return torch.tensor(s_lst, dtype=torch.float).view(batch_size, chunk_size, n_agents, obs_size).cuda(), \
        #        torch.tensor(a_lst, dtype=torch.float).view(batch_size, chunk_size, n_agents).cuda(), \
        #        torch.tensor(r_lst, dtype=torch.float).view(batch_size, chunk_size, n_agents).cuda(), \
        #        torch.tensor(s_prime_lst, dtype=torch.float).view(batch_size, chunk_size, n_agents, obs_size).cuda(), \
        #        torch.tensor(done_lst, dtype=torch.float).view(batch_size, chunk_size, 1).cuda()

    def size(self):
        return len(self.buffer)