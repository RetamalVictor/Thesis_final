import argparse
import collections

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class QNet(nn.Module):
    def __init__(self, observation_space, action_space, recurrent=False):
        super(QNet, self).__init__()
        self.num_agents = len(observation_space)
        self.recurrent = recurrent
        self.hx_size = 32
        for agent_i in range(self.num_agents):
            n_obs = observation_space[agent_i].shape[0]
            setattr(self, 'agent_feature_{}'.format(agent_i), nn.Sequential(nn.Linear(n_obs, 64),
                                                                            nn.ReLU(),
                                                                            nn.Linear(64, self.hx_size),
                                                                            nn.ReLU()))
            if recurrent:
                setattr(self, 'agent_gru_{}'.format(agent_i), nn.GRUCell(self.hx_size, self.hx_size))
            setattr(self, 'agent_q_{}'.format(agent_i), nn.Linear(self.hx_size, action_space[agent_i].n))

    def forward(self, obs, hidden):
        q_values = [torch.empty(obs.shape[0], )] * self.num_agents
        next_hidden = [torch.empty(obs.shape[0], 1, self.hx_size)] * self.num_agents
        for agent_i in range(self.num_agents):
            x = getattr(self, 'agent_feature_{}'.format(agent_i))(obs[:, agent_i, :])
            if self.recurrent:
                x = getattr(self, 'agent_gru_{}'.format(agent_i))(x, hidden[:, agent_i, :])
                next_hidden[agent_i] = x.unsqueeze(1)
            q_values[agent_i] = getattr(self, 'agent_q_{}'.format(agent_i))(x).unsqueeze(1)

        return torch.cat(q_values, dim=1), torch.cat(next_hidden, dim=1)

    def load_params(self, checkpoint, agent_i=0):
        params = torch.load(checkpoint)
        new_params = {}

        # select all keys for that have 0 in the key
        agent_keys = [key for key in params.keys() if f"_{agent_i}." in key]
        for agent in range(self.num_agents):
            for key in agent_keys:
                new_key = key.replace(f"_{agent_i}.", "_{}.".format(agent))
                new_params[new_key] = params[key]

        self.load_state_dict(new_params)

    def sample_action(self, obs, hidden, epsilon):
        out, hidden = self.forward(obs, hidden)
        mask = (torch.rand((out.shape[0],)) <= epsilon).cuda()
        action = torch.empty((out.shape[0], out.shape[1],)).cuda()
        action[mask] = torch.randint(0, out.shape[2], action[mask].shape).cuda().float()
        action[~mask] = out[~mask].argmax(dim=2).float()
        return action, hidden

    def init_hidden(self, batch_size=1):
        return torch.zeros((batch_size, self.num_agents, self.hx_size)).cuda()