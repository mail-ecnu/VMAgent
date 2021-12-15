from config import Config
import os
import copy
import numpy as np
from schedgym.sched_env import SchedEnv
from schedgym.mySubproc_vec_env import SubprocVecEnv
from utils.rl_utils import linear_decay, time_format
import argparse
from controllers import REGISTRY as mac_REGISTRY
from learners import REGISTRY as le_REGISTRY
from components import REGISTRY as mem_REGISTRY
from runx.logx import logx
from hashlib import sha1
import time


DATA_PATH = 'vmagent/data/Huawei-East-1.csv'

parser = argparse.ArgumentParser(description='Sched More Servers')
parser.add_argument('--env', type=str)
parser.add_argument('--alg', type=str)
parser.add_argument('--gamma', type=float)
parser.add_argument('--lr', type=float)
conf = parser.parse_args()
args = Config(conf.env, conf.alg)
if conf.gamma is not None:
    args.gamma = conf.gamma 
if conf.lr is not None: 
    args.lr = conf.lr

MAX_EPOCH = args.max_epoch
BATCH_SIZE = args.batch_size

logpath = '../log/search_'+str(args.learner)+conf.env+'/'+str(args.learner) + \
    str(args.gamma)+'_' + str(args.lr)+'/'

# reward discount
logx.initialize(logdir=logpath, coolname=True, tensorboard=True)


def make_env(N, cpu, mem, allow_release, double_thr=1e10):
    def _init():
        env = SchedEnv(N, cpu, mem, DATA_PATH, render_path=None,
                       allow_release=allow_release, double_thr=double_thr)
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
        if all(~alives):
            return tot_reward.mean(), tot_lenth.mean()

        avail = envs.get_attr('avail')
        feat = envs.get_attr('req')
        obs = envs.get_attr('obs')
        state = {'obs': obs, 'feat': feat, 'avail': avail}
        
        action = mac.select_actions(state, eps)
        
        action, next_obs, reward, done = envs.step(action)
        stop_idxs[alives] += 1

        next_avail = envs.get_attr('avail')
        next_feat = envs.get_attr('req')
        tot_reward[alives] += reward
        tot_lenth[alives] += 1

        buf = {'obs': obs, 'feat': feat, 'avail': avail, 'action': action,
               'reward': reward, 'next_obs': next_obs, 'next_feat': next_feat,
               'next_avail': next_avail, 'done': done}
        mem.push(buf)


if __name__ == "__main__":
    # execution
    step_list = []
    for i in range(args.num_process):
        step_list.append(np.random.randint(1000, 9999))
    my_steps = np.array(step_list)

    if args.double_thr is None:
        double_thr = 1000
    else:
        double_thr = args.double_thr

    envs = SubprocVecEnv([make_env(args.N, args.cpu, args.mem, allow_release=(
        args.allow_release == 'True'), double_thr=double_thr) for i in range(args.num_process)])

    mac = mac_REGISTRY[args.mac](args)
    print(f'Sampling with {args.mac} for {MAX_EPOCH} epochs; Learn with {args.learner}')
    learner = le_REGISTRY[args.learner](mac, args)
    mem = mem_REGISTRY[args.memory](args)
    t_start = time.time()
    for x in range(MAX_EPOCH):
        eps = linear_decay(x, [0, int(
            MAX_EPOCH * 0.25),  int(MAX_EPOCH * 0.9), MAX_EPOCH], [0.9, 0.5, 0.2, 0.2])
        envs.reset(my_steps)

        train_rew, train_len = run(
            envs, my_steps, mac, mem, learner, eps, args)
        actor_loss, critic_loss, critic1_loss, critic2_loss, alpha_loss = [0 for i in range(5)]

        # start optimization
        for i in range(args.train_n):
            batch = mem.sample(BATCH_SIZE)
            metrics = learner.train(batch)

        # log training curves
        metrics['eps'] = eps
        metrics['tot_reward'] = train_rew.mean()
        metrics['tot_len'] = train_len.mean()
        logx.metric('train', metrics, x)

        if x % args.test_interval == 0:
            envs.reset(my_steps)
            val_return, val_lenth = run(
                envs, my_steps, mac, mem, learner, 0, args)
            val_metric = {
                'tot_reward': val_return.mean(),
                'tot_len': val_lenth.mean(),
            }

            logx.metric('val', val_metric, x)

            path = f'{logpath}/models/{args.N}server-{x}'
            

            if not os.path.exists(path):
                os.makedirs(path)

            learner.save_models(path)

            t_end = time.time()
            print(f'Epoch {x}/{MAX_EPOCH}; lasted %d hour, %d min, %d sec ' %
                  time_format(t_end - t_start))
            # print('remain %d hour, %d min, %d sec' % time_format(
                # (MAX_EPOCH-x)//args.test_interval * (t_end - t_start)))
            # t_start = t_end