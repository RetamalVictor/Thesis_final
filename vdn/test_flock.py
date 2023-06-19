import torch 
import numpy as np
import torch.nn as nn

from gym_flock_uw_discrete import MultiAgentEnv
from vdn.net import QNet
from vdn.utils import ReplayBufferVDN

import argparse

def test_loaded_models(q_dir, recurrent, render=True):
    env = MultiAgentEnv(agents=15, k=4, range_start=[0,50], collision_distance=0.5, sensor_range=14)
    q = QNet(env.observation_space, env.action_space, True).cuda()
    # params = torch.load(q_dir)
    # q.load_state_dict(torch.load(q_dir))
    # scores = np.zeros(8)
    # for i in range(8):
    #     q.load_params(q_dir,agent_i=i)
    #     print(f"params for agent {i} loaded -----")
    #     test_score = test(env, 5, q, render=False)
    #     scores[i] = test_score
    #     print("test score: {:.1f}".format(test_score.item()))
    # print("mean score: {:.1f}".format(scores.mean()))
    # best_agent = np.argmax(scores)
    best_agent = 6
    print("best agent: ", best_agent)
    q.load_params(q_dir,agent_i=best_agent)
    print("params for best agent loaded -----")
    test_score = test(env, 5, q, render=render)
    # print("mean score: {:.1f}".format(scores.mean()))

def test(env, num_episodes, q, render=True):
    score = 0
    print("testing")
    for episode_i in range(num_episodes):
        state = env.reset()
        done = [False for _ in range(env.num_particles)]
        with torch.no_grad():
            hidden = q.init_hidden()
            nb_steps = 0
            while not done[1] and nb_steps < 2500:
                action, hidden = q.sample_action(torch.Tensor(state).unsqueeze(0), hidden, epsilon=0)
                action = action[0].data
                next_state, reward, done, info = env.step(action, 0.2)
                # print("reward: ", reward)
                score += sum(reward)
                state = next_state
                nb_steps += 1
                if render:
                    env.render()
                if done[1]:
                    print("Collision")
                    # break
    return score / num_episodes

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--q_dir', type=str, default=r"vdn_experiments\checkpoints\VDN_uw_c0_a\2023-02-20_13-59-06\q_1700.pth")
    parser.add_argument('--q_target_dir', type=str, default="q_target_200.pth")
    parser.add_argument('--recurrent', action='store_true')
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    test_loaded_models(args.q_dir, True, True)

    # # model = QNet(env.observation_space, env.action_space, True).cuda()
    # params = torch.load(args.q_dir)
    # # print(params.keys())
    # print(type(params))
    # new_params = {}
    # # select all keys for that have 0 in the key
    # agent_keys = [key for key in params.keys() if "_0." in key]
    # for agent in range(10):
    #     for key in agent_keys:
    #         new_key = key.replace("_0.", "_{}.".format(agent))
    #         new_params[new_key] = params[key]
    # print(new_params.keys())

