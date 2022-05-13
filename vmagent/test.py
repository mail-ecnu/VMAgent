from config import Config
import os
import copy
import numpy as np
import torch as th
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
import csv
import random
from line_profiler import LineProfiler
from torch.distributions import Categorical


parser = argparse.ArgumentParser(description='Sched More Servers')
parser.add_argument('--env', type=str,default='recovering')
parser.add_argument('--alg', type=str,default='dqn_ep_modify')
parser.add_argument('--gamma', type=float,default=0.85)
parser.add_argument('--lr', type=float,default=5e-4)
parser.add_argument('--epoch', type=int,default=3000)
parser.add_argument('--entropy', type=float,default=0.001)
parser.add_argument('--tau', type=float,default=0.01)
parser.add_argument('--rew_fn', type=str, default='cpu_allo')
parser.add_argument('--memory', type=str, default='replay')
parser.add_argument('--topk', type=int, default=5)
parser.add_argument('--num_process', type=int, default=1)
parser.add_argument('--eps2', type=float, default=0.6)
parser.add_argument('--capacity', type=int, default=1000000)
parser.add_argument('--train_n', type=int, default=5)
parser.add_argument('--N', type=int, default=5)

conf = parser.parse_args()
args = Config(conf.env, conf.alg)

DATA_PATH = 'data/dataset_deal2.csv'

args.N = conf.N
args.train_n = conf.train_n
args.capacity = conf.capacity
args.gamma = conf.gamma
args.lr = conf.lr
args.tau = conf.tau
args.entropy_coef = conf.entropy
args.memory = conf.memory
args.topk = conf.topk
args.rew_fn = conf.rew_fn
args.num_process = conf.num_process
args.eps2 = conf.eps2

MAX_EPOCH = conf.epoch
BATCH_SIZE = args.batch_size
REW_FN = conf.rew_fn

my_list = [23500]

EP_R_DIC = {}
EP_A_DIC = {}

logpath = f'mylogs/server{args.N}/'

logx.initialize(logdir=logpath, coolname=True, tensorboard=False, hparams=vars(args))


def make_env(N, cpu, mem, allow_release, double_thr=1e10):
    def _init():
        env = SchedEnv(N, cpu, mem, DATA_PATH,render_path=None, rew_fn=REW_FN, 
                       allow_release=allow_release, double_thr=double_thr, topk=args.topk)
        # env.seed(seed + rank)
        return env
    # set_global_seeds(seed)
    return _init


def run(envs, step_list, mac, mem, learner, eps, args, x, flag, ):
    tot_reward = np.array([0. for j in range(args.num_process)])
    tot_lenth = np.array([0. for j in range(args.num_process)])
    step = 0
    stop_idxs = np.array([0 for j in range(args.num_process)])

    # 初始化后面需要的lists
    avail=[]
    feat=[]
    obs=[]
    state=[]
    TMP_STATE_LST = [[] for j in range(args.num_process)]
    TMP_ACTION_LST = [[] for j in range(args.num_process)]
    TMP_RETURN_LST = [[] for j in range(args.num_process)]
    remains = [ [] for i in range(args.num_process)]
    income = 0
    while True:
        step += 1

        old_alives = envs.get_alives().copy()
        envs.update_alives()
        alives = envs.get_alives().copy()
        if all(~alives):
            if conf.alg =='dqn_ep_modify':
                for i in range(args.num_process):
                    for j in range(len(TMP_STATE_LST[i])):
                        key = TMP_STATE_LST[i][j]
                        if key not in EP_R_DIC:
                            EP_R_DIC[key] = TMP_RETURN_LST[i][-1] - TMP_RETURN_LST[i][j]
                            EP_A_DIC[key] = TMP_ACTION_LST[i][j]
                        elif EP_R_DIC[key] < TMP_RETURN_LST[i][-1] -TMP_RETURN_LST[i][j]:
                            EP_R_DIC[key] = TMP_RETURN_LST[i][-1] -TMP_RETURN_LST[i][j]
                            EP_A_DIC[key] = TMP_ACTION_LST[i][j]
            left = np.sum(remains, axis=2).sum(1).sum(0)
            return tot_lenth,tot_reward, \
            (2*args.cpu*args.N*args.num_process-left[0])/(2*args.cpu*args.N*args.num_process),\
            (2*args.mem*args.N*args.num_process-left[1])/(2*args.mem*args.N*args.num_process)

        if step==1:
            avail = envs.get_attr('avail')
            feat = envs.get_attr('req')
            obs = envs.get_attr('obs')
            state = {'obs': obs, 'feat': feat, 'avail': avail}    

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

        state = {'obs': obs, 'feat': feat, 'avail': avail}
        action = mac.select_actions(state, flag=flag, eps=eps)

        if conf.alg == 'dqn_ep_modify':
            for j in range(action.shape[0]):
                if action[j] == -1:
                    key = sha1(obs[j]).hexdigest() + sha1(feat[j]).hexdigest()
                    if key in EP_A_DIC.keys() and avail[j][EP_A_DIC[key]]==1: 
                        action[j]=EP_A_DIC[key]
                    else:
                        action[j] = Categorical(th.from_numpy(np.float32(avail[j]))).sample()
                        
        try:
            action_after, next_obs, reward, done, info = envs.step(action)
        except:
            import pdb;pdb.set_trace()

        indexs = []
        for i in range(len(alives)):
            if alives[i] == True:
                indexs.append(i)
        for i in range(len(indexs)):
            remains[indexs[i]] = next_obs[i]

        stop_idxs[alives] += 1

        if conf.alg == 'dqn_ep_mofify':
            k = 0
            for j in range(args.num_process):
                if alives[j]: 
                    TMP_STATE_LST[j].append(sha1(obs[k]).hexdigest()+sha1(feat[k]).hexdigest())
                    TMP_ACTION_LST[j].append(action[k])
                    if len(TMP_RETURN_LST[j]) > 0:
                        TMP_RETURN_LST[j].append(reward[k]+TMP_RETURN_LST[j][-1])
                    else:
                        TMP_RETURN_LST[j].append(reward[k])
                    k += 1

        next_avail = info['avail']
        next_feat = info['feat']

        tot_reward[alives] += reward
        tot_lenth[alives] += 1

        buf = {'obs': obs, 'feat': feat, 'avail': avail, 'action': action,
               'reward': reward, 'next_obs': next_obs, 'next_feat': next_feat,
               'next_avail': next_avail, 'done': done}
        mem.push(buf)
        
        avail = next_avail
        feat = next_feat
        obs = info['obs']

        
        


if __name__ == "__main__":
    if args.double_thr is None:
        double_thr = 1000
    else:
        double_thr = args.double_thr

    envs = SubprocVecEnv([make_env(args.N, args.cpu, args.mem, allow_release=(
        args.allow_release == True), double_thr=double_thr) for i in range(args.num_process)])

    mac = mac_REGISTRY[args.mac](args)
    print(f'Sampling with {args.mac} for {MAX_EPOCH} epochs; Learn with {args.learner}')
    learner = le_REGISTRY[args.learner](mac, args)
    learner.cuda()
    mem = mem_REGISTRY[args.memory](args)
    
    # model path
    path = 'models'
    learner.load_models(path,2950)

    t_start = time.time()
     

    train_steps = [23500]
    train_steps = np.array(train_steps)

    logx.msg("RL")
    write_path = f'server{conf.N}.csv'
    csv_write = csv.writer(open(write_path,'a'))
    csv_write.writerow([conf.alg])
    envs.reset(train_steps)

    train_len, train_rew, cpu_rate, mem_rate = run(
            envs, train_steps, mac, mem, learner, 0, args,0,flag=True)
    logx.msg(f"{train_len}")
    logx.msg(f"{train_len.mean()}")

    csv_write.writerow([path])

    train_len = np.array(train_len)
    train_rew = np.array(train_rew)
    csv_write.writerow(['length'])
    for i in train_len:
        csv_write.writerow([i])
    csv_write.writerow([train_len.mean()])
    csv_write.writerow(['cpu_allo'])
    for i in train_rew:
        csv_write.writerow([i])
    csv_write.writerow(['ave_cpu_allo'])
    for i in range(len(train_len)):
        csv_write.writerow([train_rew[i]/train_len[i]])
