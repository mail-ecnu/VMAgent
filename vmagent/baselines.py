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
import pickle


print(torch.cuda.is_available())

DATA_PATH = 'data/dataset_deal2.csv'
parser = argparse.ArgumentParser(description='Scexithed More Servers')

parser.add_argument('--env', type=str, default='recovering')
parser.add_argument('--baseline', type=str)
parser.add_argument('--N', type=int, default=5)
parser.add_argument('--max_epoch', type=int,default=4000)
parser.add_argument('--topk', type=int, default=50)
parser.add_argument('--rew', type=str, default='iden')

conf = parser.parse_args()
args = Config(conf.env, None)
args.N = conf.N
args.baseline = conf.baseline


logpath = f'mylogs/baselines/'+ f'{conf.env}/{str(conf.baseline)}/'

logx.initialize(logdir=logpath, coolname=True, tensorboard=True)

def make_env(N, cpu, mem, allow_release, double_thr=1e10, render_path=None):
    def _init():
        env = SchedEnv(N, cpu, mem, DATA_PATH,render_path=render_path,rew_fn=conf.rew, 
                       allow_release=allow_release, double_thr=args.double_thr, topk=conf.topk)
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
    income = 0
    avail=[]
    feat=[]
    obs=[]
    state=[]
    while True:
        step += 1
        old_alives = envs.get_alives().copy()
        envs.update_alives()
        alives = envs.get_alives().copy()
        start = time.time()
        if all(~alives):
            left = np.sum(remains, axis=2).sum(1).sum(0)
            print(tot_lenth)
            return tot_reward,tot_lenth, \
            (2*args.cpu*args.N*args.num_process-left[0])/(2*args.cpu*args.N*args.num_process),\
            (2*args.mem*args.N*args.num_process-left[1])/(2*args.mem*args.N*args.num_process)
        if step==1:
            avail = envs.get_attr('avail')
            feat = envs.get_attr('req')
            obs = envs.get_attr('obs')

        if alives.copy().tolist()!=old_alives.copy().tolist():
            indexs = []
            for i in range(len(alives)):
                if old_alives[i]==True and alives[i] == True:
                    indexs.append(True)
                if old_alives[i]==True and alives[i] == False:
                    indexs.append(False)
            avail = avail[indexs]
            feat = feat[indexs]
            obs = obs[indexs] 
        
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
        try:
            action, next_obs, reward, done, info = envs.step(action)
        except:
            import pdb;pdb.set_trace()
        
        indexs = []
        for i in range(len(alives)):
            if alives[i] == True:
                indexs.append(i)
        for i in range(len(indexs)):
            remains[indexs[i]] = next_obs[i]
        stop_idxs[alives] += 1


        next_avail = info['avail']

        tot_lenth[alives] += 1
        tot_reward[alives] += reward
        avail = next_avail


if __name__ == "__main__":


    render_path = None
    envs = SubprocVecEnv([make_env(args.N, args.cpu, args.mem, allow_release=(
        args.allow_release == True), double_thr=args.double_thr, render_path =render_path) for i in range(args.num_process)])

    step_list = [23500] 
    step_list = np.array(step_list)

    results = []
    rewards = []
    cpu_rates = []
    mem_rates = []

    if conf.baseline == 'random':
        num = 30
    else:
        num = 1
    for i in range(num):
        envs.reset(step_list)
        test_rew, test_len, cpu_rate, mem_rate = sample_baselines(
            envs, step_list, args.baseline, args)  
        cpu_rates.append(cpu_rate)
        results.append(test_len)
        mem_rates.append(mem_rate)
        rewards.append(test_rew)

    path = f'search/{args.baseline}/{conf.env}/{args.N}server/'


    if not os.path.exists(path):
        os.makedirs(path)

    results = np.array(results).max(axis=0)
    rewards = np.array(rewards).max(axis=0)
    cpu_rates = np.array(cpu_rates)
    mem_rates = np.array(mem_rates) 
    write_path = f'{conf.env}_0-20w_server{conf.N}.csv'
    csv_write = csv.writer(open(write_path,'a'))
    csv_write.writerow([args.baseline])
    csv_write.writerow(['length'])
    for i in results:
        csv_write.writerow([i])
    val_metric = {
                'train_len': results.mean(),
                'train_rew': rewards.mean(),
                'cpu_rates': cpu_rates.mean(),
                'mem_rates': mem_rates.mean(),
            }
    for x in range(conf.max_epoch):
        if x % args.test_interval == 0:
            logx.metric('val', val_metric, x)

    data = [[results.mean(),results.std()],[cpu_rates.mean(),cpu_rates.std()],[mem_rates.mean(),mem_rates.std()]]
    
    
