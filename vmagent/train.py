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
import csv
import random
from line_profiler import LineProfiler

# DATA_PATH = 'data/dataset.csv'
DATA_PATH = 'data/dataset_deal.csv'

parser = argparse.ArgumentParser(description='Sched More Servers')
parser.add_argument('--env', type=str)
parser.add_argument('--alg', type=str)
parser.add_argument('--gamma', type=float)
parser.add_argument('--lr', type=float)
parser.add_argument('--max_epoch', type=int,default=2000)
parser.add_argument('--eps', nargs='+',default=[0.8,0.6,0.3,0.1,0.1])
conf = parser.parse_args()
args = Config(conf.env, conf.alg)
args.gamma = conf.gamma
args.lr = conf.lr
args.eps = conf.eps

MAX_EPOCH = conf.max_epoch
BATCH_SIZE = args.batch_size

logpath = f'./mylogs/10step-epoch{str(MAX_EPOCH)}/'\
    + f'{conf.env}/{str(conf.alg)}/{args.eps}/{str(args.gamma)}_{str(args.lr)}'

# reward discount
logx.initialize(logdir=logpath, coolname=True, tensorboard=True)

EP_R_DIC = {}
EP_A_DIC = {}

def make_env(N, cpu, mem, allow_release, double_thr=1e10):
    def _init():
        env = SchedEnv(N, cpu, mem, DATA_PATH, render_path=None,
                       allow_release=allow_release, double_thr=double_thr)
        # env.seed(seed + rank)
        return env
    # set_global_seeds(seed)
    return _init


def run(envs, step_list, mac, mem, learner, eps, args, x, flag):
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
    while True:
        # get action
        step += 1

        old_alives = envs.get_alives().copy()
        envs.update_alives()
        alives = envs.get_alives().copy()
        if all(~alives):
            if conf.alg =='dqn_ep':
                for i in range(args.num_process):
                    for j in range(len(TMP_STATE_LST[i])):
                        key = TMP_STATE_LST[i][j]
                        if key not in EP_R_DIC:
                            EP_R_DIC[key] = TMP_RETURN_LST[i][-1] - TMP_RETURN_LST[i][j]
                            EP_A_DIC[key] = TMP_ACTION_LST[i][j]
                        elif EP_R_DIC[key] < TMP_RETURN_LST[i][-1] -TMP_RETURN_LST[i][j]:
                            EP_R_DIC[key] = TMP_RETURN_LST[i][j]
                            EP_A_DIC[key] = TMP_ACTION_LST[i][j]
            left = np.sum(remains, axis=2).sum(1).sum(0)
            return tot_reward.mean(), \
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

        if conf.alg == 'dqn_ep':
            for j in range(action.shape[0]):
                if action[j] == -1:
                    key = sha1(obs[j]).hexdigest() + sha1(feat[j]).hexdigest()
                    if key in EP_A_DIC.keys(): 
                        action[j] = EP_A_DIC[key]
        
        action, next_obs, reward, done, info = envs.step(action)

        indexs = []
        for i in range(len(alives)):
            if alives[i] == True:
                indexs.append(i)
        for i in range(len(indexs)):
            remains[indexs[i]] = next_obs[i]

        stop_idxs[alives] += 1

        if conf.alg == 'dqn_ep':
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
    t_start = time.time()

    my_list = []
    f = csv.reader(open('search.csv','r'))
    for item in f:
        my_list = item
    for i in range(len(my_list)):
        my_list[i] = int(my_list[i])
    
    # 只在前100个里面进行挑选
    my_list = my_list[:100]

    for x in range(MAX_EPOCH):
        eps = linear_decay(x, [0, int(
            MAX_EPOCH * 0.2),  int(MAX_EPOCH * 0.45),  int(MAX_EPOCH * 0.7), 
            MAX_EPOCH], [ float(i) for i in args.eps])
        

        train_steps = random.sample(my_list,args.num_process)
        train_steps = np.array(train_steps)
        envs.reset(train_steps)

        train_rew, cpu_rate, mem_rate = run(
            envs, train_steps, mac, mem, learner, eps, args,x,flag=False)

        actor_loss, critic_loss, critic1_loss, critic2_loss, alpha_loss = [0 for i in range(5)]

        for i in range(args.train_n):
            batch = mem.sample(BATCH_SIZE)
            metrics = learner.train(batch)

        # log training curves
        metrics['eps'] = eps
        metrics['tot_reward'] = train_rew.mean()
        logx.metric('train', metrics, x)

        if x % args.test_interval == 0:
            train_steps = np.array(my_list)

            train_rews = []
            cpu_rates = []
            mem_rates = []
            
            for i in range(10):
                train_list = train_steps[i*args.num_process : (i+1)*args.num_process]
                envs.reset(train_list)
                train_rew, cpu_rate, mem_rate = run(
                    envs, train_list, mac, mem, learner, 0, args, 0, flag=True)
                train_rews.append(train_rew)
                cpu_rates.append(cpu_rate)
                mem_rates.append(mem_rate)
                
            train_rews = np.array(train_rews)
            cpu_rates = np.array(cpu_rates)
            mem_rates = np.array(mem_rates)

            # mem.clean()
            val_metric = {
                'train_len': train_rews.mean(),
                'cpu_rates': cpu_rates.mean(),
                'mem_rates': mem_rates.mean(),
            }
            logx.metric('val', val_metric, x)
            path = 'models/random_steps'+ '/' + conf.env+ '/'+str(args.learner)+'/'+str(args.gamma)+'_' + str(args.lr)+'/' + str(args.N)+'server'

            if not os.path.exists(path):
                os.makedirs(path)

            learner.save_models(path)

            t_end = time.time()
            print(f'Epoch {x}/{MAX_EPOCH}; lasted %d hour, %d min, %d sec ' %
                  time_format(t_end - t_start))
            # print('remain %d hour, %d min, %d sec' % time_format(
                # (MAX_EPOCH-x)//args.test_interval * (t_end - t_start)))
            # t_start = t_end
