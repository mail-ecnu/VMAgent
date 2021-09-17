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
import torch
import pdb
import time
import random
print(torch.cuda.is_available())

from config import Config

DATA_PATH = 'data/mydata.csv'

parser = argparse.ArgumentParser(description='Sched More Servers')
parser.add_argument('--env', type=str)
parser.add_argument('--alg', type=str)
conf = parser.parse_args()
args = Config(conf.env, conf.alg)
MAX_EPOCH = args.max_epoch
BATCH_SIZE = args.batch_size
# reward discount
logx.initialize(logdir='./log/', coolname=True, tensorboard=True)

def make_env(N, cpu, mem, allow_release):
    def _init():
        env = SchedEnv(N, cpu, mem, DATA_PATH, render_path=None,allow_release=allow_release)
        # env.seed(seed + rank)
        return env
    # set_global_seeds(seed)
    return _init


def run(envs, step_list, mac, mem, learner, eps, args):

    tot_reward = np.array([0. for j in range(args.num_process)])
    tot_lenth = np.array([0. for j in range(args.num_process)])
    step = 0
    stop_idxs = np.array([0 for j in range(args.num_process)])
    while True:
        # get action
        step += 1
        envs.update_alives()

        alives = envs.get_alives().copy()
        if  all(~alives):
            return tot_reward.mean(), tot_lenth.mean()
    
        avail = envs.get_attr('avail')
        feat = envs.get_attr('req')
        obs = envs.get_attr('obs')
        state = {'obs':obs, 'feat': feat, 'avail': avail}
        
        action = mac.select_actions(state, eps)
        action, next_obs, reward, done = envs.step(action)
        stop_idxs[alives] += 1

        next_avail = envs.get_attr('avail')
        next_feat = envs.get_attr('req')
        tot_reward[alives] += reward
        tot_lenth[alives] += 1
        
        buf = {'obs':obs,'feat':feat, 'avail':avail, 'action':action,
                'reward': reward, 'next_obs':next_obs, 'next_feat':next_feat,
                'next_avail': next_avail, 'done': done}
        mem.push(buf)


if __name__ == "__main__":


    # execution
    
    step_list = np.array([42132])
    my_steps = step_list.repeat(args.num_process)

    envs = SubprocVecEnv([make_env(args.N, args.cpu, args.mem, allow_release=(args.allow_release=='True')) for i in range(args.num_process)])
    mac = mac_REGISTRY[args.mac](args)
    learner = le_REGISTRY[args.learner](mac, args)
    mem = mem_REGISTRY[args.memory](args)

    for x in range(MAX_EPOCH):
        t_start = time.time()
        eps = linear_decay(x, [0, int(MAX_EPOCH * 0.25),  int(MAX_EPOCH * 0.9), MAX_EPOCH], [0.9, 0.5, 0.2, 0.2])
        envs.reset(my_steps)

        train_rew, train_len = run(envs, my_steps, mac, mem, learner, eps, args)
        loss = 0
        print(f'epoch {x}')
        for i in range(args.train_n):
            # learn_cnt += 1
            batch = mem.sample(BATCH_SIZE)
            start = time.time()
            loss += learner.train(batch)
        metrics = {
            'eps': eps,
            'tot_reward': train_rew.mean(),
            'tot_len': train_len.mean(),
            'loss': loss
        }
        logx.metric('train', metrics, x)

        if x % args.test_interval == 0:
            envs.reset(my_steps)
            val_return, val_lenth = run(envs, my_steps, mac, mem, learner, 0, args)
            mem.clean()
            val_metric = {
                'tot_reward': val_return.mean(),
                'tot_len': val_lenth.mean(),
            }
            
            logx.metric('val', val_metric, x)

            path = f'models/{args.N}server-{x}'

            if not os.path.exists(path):
                os.makedirs(path)

            learner.save_models(path)
            
            t_end = time.time()
            print('lasted %d hour, %d min, %d sec '% time_format(t_end - t_start))
            print('remain %d hour, %d min, %d sec'% time_format((MAX_EPOCH-x)//args.test_interval * (t_end - t_start)))
            t_start = t_end



