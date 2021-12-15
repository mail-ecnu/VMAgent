from config import Config
import os
import numpy as np
from schedgym.sched_env import SchedEnv
from schedgym.mySubproc_vec_env import SubprocVecEnv
import argparse
from controllers import REGISTRY as mac_REGISTRY
from learners import REGISTRY as le_REGISTRY
from components import REGISTRY as mem_REGISTRY
from runx.logx import logx
from hashlib import sha1
import torch
import pandas as pd
import pdb
import time
import csv


DATA_PATH = 'vmagent/data/Huawei-East-1.csv'
parser = argparse.ArgumentParser(description='Sched More Servers')

parser.add_argument('--env', type=str)
parser.add_argument('--baseline', type=str)
parser.add_argument('--N', type=int)
conf = parser.parse_args()
args = Config(conf.env, None)
args.N = conf.N
args.baseline = conf.baseline
args.num_process = 50 
def make_env(N, cpu, mem, allow_release, double_thr=1e10):
    def _init():
        env = SchedEnv(N, cpu, mem, DATA_PATH, render_path=None,
                       allow_release=allow_release, double_thr=double_thr)
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
    last_obs = []
    while True:
        step += 1
        envs.update_alives()

        alives = envs.get_alives().copy()
        start = time.time()
        if all(~alives):
            last_obs = np.array(last_obs)
            remains = np.sum(last_obs, axis=2).sum(1)
            return tot_lenth.mean(), (2*args.cpu*args.num_process-remains[0][0])/(2*args.cpu*args.num_process),(2*args.mem*args.num_process-remains[0][1])/(2*args.mem*args.num_process)
        avail = envs.get_attr('avail')
        feat = envs.get_attr('req')
        obs = envs.get_attr('obs')
        state = {'obs': obs, 'feat': feat, 'avail': avail}
        if method == 'ff':
            action = np.ones(obs.shape[0]) * -1
        elif method == 'bf':
            action = np.ones(obs.shape[0]) * -2

        action, next_obs, reward, done = envs.step(action)
        last_obs = next_obs
        
        # print(f'next obs: {next_obs},')
        
        if (next_obs<0).sum():
            import pdb; pdb.set_trace()

        k = 0
        stop_idxs[alives] += 1

        next_avail = envs.get_attr('avail')
        next_feat = envs.get_attr('req')

        tot_lenth[alives] += 1


if __name__ == "__main__":
    render_path = f'{args.baseline}-{args.N}.p'
    envs = SubprocVecEnv([make_env(args.N, args.cpu, args.mem, allow_release=(
        args.allow_release == 'True'), double_thr=args.double_thr) for i in range(args.num_process)])

    step_list = []
    for i in range(args.num_process):
        step_list.append(np.random.randint(1000, 9999))
    step_list = np.array(step_list)
    # for i in range(1000):
    #     step_list.append(np.random.randint(0, 16000))
    # step_list = np.array(step_list)

    results = []
    cpu_rates = []
    mem_rates = []

    print(step_list[:args.num_process])
    envs.reset(step_list[:args.num_process])

    for j in range(int(step_list.size/args.num_process)):
        local_list = step_list[j:j+args.num_process]
        try:
            envs.reset(local_list)
        except:
            import pdb; pdb.set_trace()
        test_len, cpu_rate, mem_rate = sample_baselines(
            envs, local_list, args.baseline, args)
        cpu_rates.append(cpu_rate)
        results.append(test_len)
        mem_rates.append(mem_rate)
        # print(f'cpu_rate:{cpu_rate}')
        # print(f'test_len:{test_len}')
        # print(f'mem_rate:{mem_rate}')

    path = f'../logs/{args.baseline}/{conf.env}/{args.N}server/'

    if not os.path.exists(path):
        os.makedirs(path)

    results = np.array(results)
    cpu_rates = np.array(cpu_rates)
    mem_rates = np.array(mem_rates) 
    data = [[results.mean(),results.std()],[cpu_rates.mean(),cpu_rates.std()],[mem_rates.mean(),mem_rates.std()]]

    with open(path+'/results.csv','w')as f:
        f_csv = csv.writer(f)
        f_csv.writerow(step_list)
        f_csv.writerow(data)