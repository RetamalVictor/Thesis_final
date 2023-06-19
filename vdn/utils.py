
import numpy as np
import collections
from collections import deque
import torch 
import torch.nn as nn
from copy import copy

class ReplayBufferVDN(object):
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


class ReplayBuffer(object):
    """
    Needs to be modified to account for the number of agents.
    It will receive a batch of states, actions, rewards, next_states, and done
    The batch is the number of agents

    observations = [500, 62]
    
    """
    def __init__(self, max_size, input_shape, chunk_size, n_actions, n_agents):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.n_agents = n_agents
        self.state_memory = torch.zeros((self.mem_size, chunk_size, n_agents, *input_shape), dtype=torch.float32)
        self.new_state_memory = torch.zeros((self.mem_size, chunk_size, n_agents, *input_shape), dtype=torch.float32)
        self.action_memory = torch.zeros((self.mem_size, chunk_size, n_agents), dtype=torch.float32)
        self.reward_memory = torch.zeros((self.mem_size,chunk_size, n_agents), dtype=torch.float32)
        self.terminal_memory = torch.zeros(self.mem_size, chunk_size, n_agents, dtype=torch.float32)

    def store_transitions(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index:index+self.n_agents] = state
        self.new_state_memory[index:index+self.n_agents] = state_
        self.action_memory[index:index+self.n_agents] = action
        self.reward_memory[index:index+self.n_agents] = reward
        self.terminal_memory[index:index+self.n_agents] = 1 - done
        self.mem_cntr += self.n_agents

    def store_single_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index:index+self.n_agents] = state
        self.new_state_memory[index:index+self.n_agents] = state_
        self.action_memory[index:index+self.n_agents] = action
        self.reward_memory[index:index+self.n_agents] = reward
        self.terminal_memory[index:index+self.n_agents] = 1 - done
        self.mem_cntr += self.n_agents
        
    def sample_buffer(self, batch_size):

        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states.cuda(), actions.cuda(), rewards.cuda(), states_.cuda(), terminal.cuda()
    



class PrioritizedReplayVDN(object):
    def __init__(self, buffer_limit, chunk_size, n_agents, input_shape, batch_size=128, seed=0, gamma=0.99, n_step=100, alpha=0.6, beta_start = 0.4, beta_frames=100000):
        self.buffer = collections.deque(maxlen=buffer_limit)

        self.state_memory = torch.zeros((batch_size, chunk_size, n_agents, *input_shape), dtype=torch.float32).cuda()
        self.new_state_memory = torch.zeros((batch_size, chunk_size, n_agents, *input_shape), dtype=torch.float32).cuda()
        self.action_memory = torch.zeros((batch_size, chunk_size, n_agents), dtype=torch.float32).cuda()
        self.reward_memory = torch.zeros((batch_size,chunk_size, n_agents), dtype=torch.float32).cuda()
        self.terminal_memory = torch.zeros((batch_size, chunk_size, 1), dtype=torch.float32).cuda()

        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1 
        self.batch_size = batch_size
        self.capacity = buffer_limit
        self.pos = 0
        self.priorities = np.zeros((buffer_limit,), dtype=np.float32)
        self.seed = np.random.seed(seed)
        self.n_step = n_step
        self.n_step_buffer = deque(maxlen=self.n_step)
        self.gamma = gamma

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
        self.n_step_buffer.append(transition)
        if len(self.n_step_buffer) == self.n_step:
            s, a, r, s_prime, done = self.calc_multistep_return()

        max_prio = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append((s, a, r, s_prime, done))
        else:
            self.buffer[self.pos] = (s, a, r, s_prime, done)
        
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def calc_multistep_return(self):
        Return = 0
        for idx in range(self.n_step):
            Return += self.gamma**idx * self.n_step_buffer[idx][2]
        
        return self.n_step_buffer[0][0], self.n_step_buffer[0][1], Return, self.n_step_buffer[-1][3], self.n_step_buffer[-1][4]

    def beta_by_frame(self, frame_idx):
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)

    def sample_chunk(self, batch_size, chunk_size):
        N = len(self.buffer)
        if N == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        # calculate P = p^a/sum(p^a)
        probs  = prios ** self.alpha
        P = probs / probs.sum()

        # get the indices depending on the probability p
        start_idx = np.random.choice(N, batch_size, p=P)

        beta = self.beta_by_frame(self.frame)
        self.frame += 1

        # Compute importance-sampling weight
        weights  = (N * P[start_idx]) ** (-beta)
        # normalize weights
        weights /= weights.max() 
        weights  = np.array(weights, dtype=np.float32) 

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
            batch += 1

        return self.state_memory.clone(), \
                self.action_memory.clone(), \
                self.reward_memory.clone(), \
                self.new_state_memory.clone(), \
                self.terminal_memory.clone(), \
                start_idx, \
                weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio 

    def size(self):
        return len(self.buffer)

    