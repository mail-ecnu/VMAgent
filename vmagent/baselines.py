from config import Config
import os
import copy
import numpy as np
from schedgym.sched_env import SchedEnv
from schedgym.mySubproc_vec_env import SubprocVecEnv
from utils.rl_utils import linear_decay, time_format
import argparse
from components.replay_memory import ReplayMemory
from controllers import REGISTRY as mac_REGISTRY
from learners import REGISTRY as le_REGISTRY
from components import REGISTRY as mem_REGISTRY
from runx.logx import logx
from hashlib import sha1
import torch
import pandas as pd
import pdb
import time
import random
import csv


print(torch.cuda.is_available())

# DATA_PATH = 'data/dataset.csv'
DATA_PATH = 'data/dataset_deal.csv'
parser = argparse.ArgumentParser(description='Sched More Servers')

parser.add_argument('--env', type=str)
parser.add_argument('--baseline', type=str)
parser.add_argument('--N', type=int)
parser.add_argument('--max_epoch', type=int,default=4000)
conf = parser.parse_args()
args = Config(conf.env, None)
args.N = conf.N
args.baseline = conf.baseline

logpath = f'./mylogs/10step-epoch{str(conf.max_epoch)}/'\
    + f'{conf.env}/{str(conf.baseline)}/'

logx.initialize(logdir=logpath, coolname=True, tensorboard=True)

def make_env(N, cpu, mem, allow_release, double_thr=1e10, render_path=None):
    def _init():
        env = SchedEnv(N, cpu, mem, DATA_PATH, render_path=render_path,
                       allow_release=allow_release, double_thr=args.double_thr)
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
    remains = [ [] for i in range(args.num_process)]
    while True:
        step += 1
        envs.update_alives()

        alives = envs.get_alives().copy()
        start = time.time()
        if all(~alives):
            left = np.sum(remains, axis=2).sum(1).sum(0)
            print(tot_lenth)
            print((2*args.cpu*args.N*args.num_process-left[0])/(2*args.cpu*args.N*args.num_process))
            print((2*args.mem*args.N*args.num_process-left[1])/(2*args.mem*args.N*args.num_process))
            return tot_lenth.mean(), \
            (2*args.cpu*args.N*args.num_process-left[0])/(2*args.cpu*args.N*args.num_process),\
            (2*args.mem*args.N*args.num_process-left[1])/(2*args.mem*args.N*args.num_process)
        avail = envs.get_attr('avail')
        feat = envs.get_attr('req')
        obs = envs.get_attr('obs')
        state = {'obs': obs, 'feat': feat, 'avail': avail}
        if method == 'ff':
            action = np.ones(obs.shape[0]) * -1
        elif method == 'bf_cpu':
            action = np.ones(obs.shape[0]) * -2
        elif method == 'bf_mem':
            action = np.ones(obs.shape[0]) * -3
        elif method == 'random':
            action = np.zeros(obs.shape[0],dtype=int)
            for j in range(len(avail)):
                indexs = []
                for i in range(len(avail[j])):
                    if avail[j][i] == True:
                        indexs.append(i)
                action[j] = random.sample(indexs,1)[0]

        action, next_obs, reward, done, info = envs.step(action)
        
        indexs = []
        for i in range(len(alives)):
            if alives[i] == True:
                indexs.append(i)
        for i in range(len(indexs)):
            remains[indexs[i]] = next_obs[i]
        
        # print(f'next obs: {next_obs},')
        
        if (next_obs<0).sum():
            import pdb; pdb.set_trace()

        k = 0
        stop_idxs[alives] += 1

        next_avail = envs.get_attr('avail')
        next_feat = envs.get_attr('req')
        # print(f'next avial: {next_avail}')
        # print(f'next feat: {next_feat}')

        tot_lenth[alives] += 1


if __name__ == "__main__":

    render_path = f'render/{args.baseline}/{args.baseline}-{args.N}.p'
    envs = SubprocVecEnv([make_env(args.N, args.cpu, args.mem, allow_release=(
        args.allow_release == True), double_thr=args.double_thr, render_path =None) for i in range(args.num_process)])

    step_list = []
    f = csv.reader(open('search.csv','r'))
    for item in f:
        step_list = item
    for i in range(len(step_list)):
        step_list[i] = int(step_list[i])
    step_list = np.array(step_list)
    # for i in range(1000):
    #     step_list.append(np.random.randint(0, 16000))
    # step_list = np.array(step_list)

    results = []
    cpu_rates = []
    mem_rates = []

    for i in range(10):
        val_list = step_list[i*args.num_process : (i+1)*args.num_process]
        envs.reset(val_list)
        test_len, cpu_rate, mem_rate = sample_baselines(
            envs, val_list, args.baseline, args)  
        cpu_rates.append(cpu_rate)
        results.append(test_len)
        mem_rates.append(mem_rate)

    path = f'search/{args.baseline}/{conf.env}/{args.N}server/'

    if not os.path.exists(path):
        os.makedirs(path)

    results = np.array(results)
    cpu_rates = np.array(cpu_rates)
    mem_rates = np.array(mem_rates) 
    val_metric = {
                'train_len': results.mean(),
                'cpu_rates': cpu_rates.mean(),
                'mem_rates': mem_rates.mean(),
            }
    for x in range(conf.max_epoch):
        if x % args.test_interval == 0:
            logx.metric('val', val_metric, x)

    data = [[results.mean(),results.std()],[cpu_rates.mean(),cpu_rates.std()],[mem_rates.mean(),mem_rates.std()]]
