import os
import copy
import numpy as np
from gym_sched.envs.sched_env import  SchedEnv
from gym_sched.envs.mySubproc_vec_env import SubprocVecEnv
from utils.rl_utils import linear_decay, time_format
import argparse
from components.replay_memory import ReplayMemory
from controllers import REGISTRY as mac_REGISTRY
from learners import REGISTRY as le_REGISTRY
from components import REGISTRY as mem_REGISTRY
from runx.logx import logx
from hashlib import sha1
import pandas as pd
import pdb
import time
import random

from config import Config

DATA_PATH = 'vmagent/data/dataset.csv'
parser = argparse.ArgumentParser(description='Sched More Servers')

parser.add_argument('--env', type=str)
parser.add_argument('--baseline', type=str)
conf = parser.parse_args()
args = Config(conf.env, None)
args.baseline = conf.baseline

def make_env(N, cpu, mem, allow_release):
    def _init():
        env = SchedEnv(N, cpu, mem, DATA_PATH, render_path=None,allow_release=allow_release)
        # env.seed(seed + rank)
        return env
    # set_global_seeds(seed)
    return _init

def sample_baselines(envs, step_list, method, args):
    tot_reward = np.array([0. for j in range(args.num_process)])
    tot_lenth = np.array([0. for j in range(args.num_process)])
    step = 0
    nstep_buf_list = []
    stop_idxs = np.array([0 for j in range(args.num_process)])
    loss = 0
    learn_cnt = 1
    while True:
        step += 1
        envs.update_alives()

        alives = envs.get_alives().copy()
        start = time.time()
        if  all(~alives):
            return tot_lenth.mean()
        avail = envs.get_attr('avail')
        feat = envs.get_attr('req')
        obs = envs.get_attr('obs')
        state = {'obs':obs, 'feat': feat, 'avail': avail}
        if method == 'ff':
            action = np.ones(obs.shape[0]) * -1
        elif method == 'bf':
            action = np.ones(obs.shape[0]) * -2

        action, next_obs, reward, done = envs.step(action)

        k = 0
        stop_idxs[alives] += 1

        next_avail = envs.get_attr('avail')
        next_feat = envs.get_attr('req')
        tot_lenth[alives] += 1



if __name__ == "__main__":

    render_path = f'{args.baseline}-{args.N}.p'
    args.num_process = 1
    envs = SubprocVecEnv([make_env(args.N, args.cpu, args.mem, allow_release=(args.allow_release=='True')) for i in range(args.num_process)])

    step_list = [1234]
    results = {}

    envs.reset(step_list)
    test_len = sample_baselines(envs, step_list, args.baseline, args)
    results[args.baseline] = test_len

    print(results)
