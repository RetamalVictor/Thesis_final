import os, time, torch, yaml
import numpy as np
from tqdm import tqdm
import argparse
import sys
sys.path.append("./environments/")

from utils import ReplayBufferVDN
from net import QNet
from gym_flock_uw_discrete import MultiAgentEnv

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


def train(q, q_target, memory, optimizer, gamma, batch_size, update_iter=10, chunk_size=10, grad_clip_norm=5):
    _chunk_size = chunk_size if q.recurrent else 1
    for _ in range(update_iter):
        s, a, r, s_prime, done = memory.sample_chunk(batch_size, _chunk_size)

        hidden = q.init_hidden(batch_size)
        target_hidden = q_target.init_hidden(batch_size)
        loss = 0
        for step_i in range(_chunk_size):
            q_out, hidden = q(s[:, step_i, :, :], hidden)
            q_a = q_out.gather(2, a[:, step_i, :].unsqueeze(-1).long()).squeeze(-1)
            sum_q = q_a.sum(dim=1, keepdims=True)

            max_q_prime, target_hidden = q_target(s_prime[:, step_i, :, :], target_hidden.detach())
            max_q_prime = max_q_prime.max(dim=2)[0].squeeze(-1)
            target_q = r[:, step_i, :].sum(dim=1, keepdims=True)
            target_q += gamma * max_q_prime.sum(dim=1, keepdims=True) * (1 - done[:, step_i]) # may be faulty

            loss += F.smooth_l1_loss(sum_q, target_q.detach())

            done_mask = done[:, step_i].squeeze(-1).bool()
            hidden[done_mask] = q.init_hidden(len(hidden[done_mask]))
            target_hidden[done_mask] = q_target.init_hidden(len(target_hidden[done_mask]))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(q.parameters(), grad_clip_norm, norm_type=2)
        optimizer.step()

def test(env, num_episodes, q, render=False):
    score = 0
    for episode_i in range(num_episodes):
        state = env.reset()
        done = [False for _ in range(env.num_particles)]
        with torch.no_grad():
            hidden = q.init_hidden()
            nb_steps = 0
            while not done[1] and nb_steps < 150:
                action, hidden = q.sample_action(torch.Tensor(state).unsqueeze(0), hidden, epsilon=0)
                action = action[0].data
                next_state, reward, done, info = env.step(action)
                score += sum(reward)
                state = next_state
                nb_steps += 1
                if render:
                    env.render()
    return score / num_episodes

def test_loaded_models(q_dir, q_target_dir, recurrent, render=True):
    env = MultiAgentEnv(agents=12, k=4, range_start=[0,100], sensor_range=14)
    q = QNet(env.observation_space, env.action_space, recurrent).cuda()
    q_target = QNet(env.observation_space, env.action_space, recurrent).cuda()
    q.load_state_dict(torch.load(q_dir))
    q_target.load_state_dict(torch.load(q_target_dir))
    test_score = test(env, 5, q)
    print("test score: {:.1f}".format(test_score))

def main(exp_name, lr, gamma, batch_size, buffer_limit, log_interval, checkpoint, max_episodes, max_epsilon, min_epsilon,
         test_episodes, warm_up_steps, update_iter, chunk_size, update_target_interval, recurrent):

    writer = SummaryWriter(log_dir=os.path.join(r"vdn_experiments\runs", exp_name, time.strftime("%Y-%m-%d_%H-%M-%S")))
    # create env.
    env = MultiAgentEnv(agents=8, k=4, range_start=[0,50])
    test_env = MultiAgentEnv(agents=8, k=4, range_start=[0,50])
    memory = ReplayBufferVDN(buffer_limit, chunk_size=10, n_agents=env.num_particles, input_shape=[env.k])

    # create networks
    q = QNet(env.observation_space, env.action_space, recurrent).cuda()
    q_target = QNet(env.observation_space, env.action_space, recurrent).cuda()
    q_target.load_state_dict(q.state_dict())
    optimizer = optim.Adam(q.parameters(), lr=lr)

    checkpoint = os.path.join(checkpoint, exp_name, time.strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(checkpoint, exist_ok=True)
    score = 0
    for episode_i in range(max_episodes):
        epsilon = max(min_epsilon, max_epsilon - (max_epsilon - min_epsilon) * (episode_i / (0.6 * max_episodes)))
        state = env.reset()
        done = [False for _ in range(env.num_particles)]
        nb_steps = 0
        with torch.no_grad():
            hidden = q.init_hidden()
            while not done[1] and nb_steps < 150:
                action, hidden = q.sample_action(torch.Tensor(state).unsqueeze(0), hidden, epsilon)
                action = action[0].data
                next_state, reward, done, info = env.step(action)
                memory.put((state, action, reward, next_state, [int(done[1])]))
                score += sum(reward)
                state = next_state
                nb_steps += 1

        if memory.size() > warm_up_steps:
            train(q, q_target, memory, optimizer, gamma, batch_size, update_iter, chunk_size)

        writer.add_scalar('train/score', score, episode_i)
        writer.add_scalar('train/epsilon', epsilon, episode_i)
        writer.add_scalar('train/buffer_size', memory.size(), episode_i)

        if episode_i % update_target_interval:
            q_target.load_state_dict(q.state_dict())

        if (episode_i + 1) % log_interval == 0:
            test_score = test(test_env, test_episodes, q)
            writer.add_scalar('test/score', test_score, episode_i)
            train_score = score / log_interval
            print(f"Episode {episode_i} finished after {nb_steps} steps\n train score: {train_score.item()} test score: {test_score.item()}")
            score = 0

        # save both models if episode modulo 1000 equals 0
        if (episode_i) % 100 == 0:
            torch.save(q.state_dict(), os.path.join(checkpoint,'q_{}.pth'.format(episode_i)))
            torch.save(q_target.state_dict(), os.path.join(checkpoint,'q_target_{}.pth'.format(episode_i)))


    env.close()
    test_env.close()


if __name__ == '__main__':
    # Lets gather arguments
    parser = argparse.ArgumentParser(description='Value Decomposition Network (VDN)')
    parser.add_argument('--exp-name', required=False, default='VDN_uw_c0_a')
    parser.add_argument('--seed', type=int, default=1, required=False)
    parser.add_argument('--no-recurrent', action='store_true')
    parser.add_argument('--max-episodes', type=int, default=15000, required=False)
    parser.add_argument('--max-epsilon', type=float, default=0.9, required=False)
    parser.add_argument('--min-epsilon', type=float, default=0.1, required=False)
    parser.add_argument('--test-episodes', type=int, default=5, required=False)
    parser.add_argument('--warm-up-steps', type=int, default=2000, required=False)
    parser.add_argument('--update-iter', type=int, default=10, required=False)
    parser.add_argument('--chunk-size', type=int, default=10, required=False)
    parser.add_argument('--update-target-interval', type=int, default=20, required=False)
    # parser.add_argument('--recurrent', action='store_true')
    parser

    # Process arguments
    args = parser.parse_args()

    kwargs = {'exp_name': args.exp_name,
              'lr': 0.001,
              'batch_size': 32,
              'gamma': 0.99,
              'buffer_limit': 50000,
              'update_target_interval': 20,
              'log_interval': 1,
              'checkpoint': r'vdn_experiments\checkpoints',
              'max_episodes': args.max_episodes,
              'max_epsilon': 0.9,
              'min_epsilon': 0.1,
              'test_episodes': 5,
              'warm_up_steps': 2000,
              'update_iter': 10,
              'chunk_size': 10,  # if not recurrent, internally, we use chunk_size of 1 and no gru cell is used.
              'recurrent': not args.no_recurrent}

    # if USE_WANDB:
    #     import wandb

    #     wandb.init(project='minimal-marl', config={'algo': 'vdn', **kwargs})

    main(**kwargs)
    # test_loaded_models('q_.pth', 'q_target_14999.pth', **kwargs)