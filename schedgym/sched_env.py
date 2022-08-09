from argparse import Action
import pandas as pd
import numpy as np
import random
import gym
from gym import spaces
import json


def getData(path, double_thr=1e10, smallFilter=False):
    '''
    csv_data: input a DataFrame Object
    return: A list like [{'uuid':'',...},{},...]
    '''
    csv_data = pd.read_csv(path)
    l = np.array(csv_data).tolist()
    index = 0
    for item in l:
        newdict = {'uuid': item[0], 'cpu': item[1], 'mem': item[2],
                   'time': item[3], 'type': int(item[4]), 'is_double': 0}
        # filter all the small size VMs
        if smallFilter and newdict['cpu']<4:
            continue
        # double NUMA threshold for cpu
        if newdict['cpu'] >= double_thr:
            newdict['is_double'] = 1
        l[index] = newdict
        index += 1
    return l

def getSData(path):
    '''
    csv_data: input a DataFrame Object
    return: A list like [{'uuid':'',...},{},...]
    '''
    csv_data = pd.read_csv(path)
    l = np.array(csv_data).tolist()
    index = 0
    res = {}
    for item in l:
        res[make_key(item[0], item[1])] = item[2]
    return res

def make_key(cpu, mem):
    return 'c'+str(int(cpu)) +'m'+str(int(mem))



class Server():
    '''
        maintain cpu, mem and stored request in state.
        and provide request handler for "create" and "delete".
    '''
    def __init__(self, cpu, mem):
        self.tot_cpu = [cpu, cpu]
        self.tot_mem = [mem, mem]
        self.remain_cpu = [cpu, cpu]
        self.remain_mem = [mem, mem]
        self.stored = [{}, {}]


    def split_req(self, request):
        '''
            split requests for double numa requests
        '''
        reqs = [{}, {}]
        for key in request.keys():
            factor = 1
            if key == 'cpu' or key == 'mem':
                factor = 2
                reqs[0][key] = request[key]/factor
                reqs[1][key] = request[key]/factor
            else:
                reqs[0][key] = request[key]
                reqs[1][key] = request[key]
        return reqs

    def _allocate(self, request, numa=0):
        self.remain_cpu[numa] -= request['cpu']
        self.remain_mem[numa] -= request['mem']
        self.stored[numa][request['uuid']] = request

    def handle(self, request, numa=0):
        '''
            handle request based on its type
        '''

        self.allocate(request, numa=numa)
        assert self.remain_cpu >= [0,0], "remain cpu should be positive"
        assert self.remain_mem >= [0,0], "remain mem should be positive"
        return True

    def allo_rate(self, numa, cpu_selection):
        '''
            calculate allocation rate after creation.
            cpu_allo_rate if cpu_selection is True.
            mem_allo_rate if cpu_selection is False.
        '''
        if cpu_selection:
            return self.remain_cpu[numa] / self.tot_cpu[numa]
        else:
            return self.remain_mem[numa] / self.tot_mem[numa]      

    def allocate(self, request, numa=0):
        '''
            allocate request on the server.
        '''
        if request['is_double'] == 1:
            reqs = self.split_req(request)
            self._allocate(reqs[0], numa=0)
            self._allocate(reqs[1], numa=1)
        else:
            self._allocate(request, numa=numa)
        return True


    def delete(self, request):
        '''
            delete request on the server.
        '''
        del_status = -1
        for i in range(2):
            stored = self.stored[i]
            if request['uuid'] in stored.keys():
                stored.pop(request['uuid'])
                if request['is_double'] == 1:
                    self.remain_cpu[i] += request['cpu']/2
                    self.remain_mem[i] += request['mem']/2
                    del_status = 0

                else:
                    self.remain_cpu[i] += request['cpu']
                    self.remain_mem[i] += request['mem']
                    del_status = i
        return del_status

    def _usable(self, request, numa):
        '''
            check if server can handle the request.
        '''
        if self.remain_mem[numa] >= request['mem'] and self.remain_cpu[numa] >= request['cpu']:
            return 1
        else:
            return 0

    def usable(self, request):
        '''
            check the request's usable attribute.
            1 is usable, 0 otherwise.
        '''
        usable = [0, 0]
        if request['is_double'] == 1:
            reqs = self.split_req(request)
            if self._usable(reqs[0], numa=0)==1 and self._usable(reqs[1], numa=1)==1:
                usable = [1, 1]
            else:
                usable = [0, 0]
        else:
            for i in range(2):
                usable[i] = self._usable(request, numa=i)
        return usable

    def reset_status(self, status):
        '''
            reset server's attribute.
        '''
        self.remain_cpu[0] = status[0][0]
        self.remain_cpu[1] = status[1][0]
        self.remain_mem[0] = status[0][1]
        self.remain_mem[1] = status[1][1]

    def describe(self):
        '''
            return remain cpu and mem
        '''
        return [[self.remain_cpu[0], self.remain_mem[0]],
                [self.remain_cpu[1], self.remain_mem[1]]]

class Cluster():
    def __init__(self, N, cpu, mem):
        self.N = N
        self.cpu = cpu
        self.mem = mem
        self.servers = [Server(self.cpu, self.mem) for i in range(self.N)]

    def handle(self, action, request):
        '''
            action[0] is server id
        '''
        self.servers[action//2].handle(request, numa=action%2)

    def first_fit_action(self, request):
        '''
            get action for first-fit
        '''
        i = -1
        for server in self.servers:
            i += 1
            usable = server.usable(request)
            for numa in range(2):
                if usable[numa] == 1:
                    return i * 2 + numa

    def best_fit_action(self, request, cpu_selection):
        '''
            get action for best-fit.
            cpu_selection defines bf for cpu or mem.
        '''
        i = -1
        bst_action = 20
        bst_allo_rate = 2
        for server in self.servers:
            i += 1
            usable = server.usable(request)
            if request['is_double'] == 1 and usable[0] == 1:
                server.handle(request)
                allo_rate = max(server.allo_rate(0,cpu_selection), server.allo_rate(1,cpu_selection))
                if allo_rate < bst_allo_rate:
                    bst_allo_rate = allo_rate
                    bst_action = i * 2
                server.delete(request)
            else:
                for numa in range(2):
                    if usable[numa] == 1:
                        server.handle(request, numa)
                        allo_rate = min(bst_allo_rate, server.allo_rate(numa,cpu_selection))
                        if allo_rate < bst_allo_rate:
                            bst_allo_rate = allo_rate
                            bst_action = i * 2 + numa
                        server.delete(request)
        return bst_action

    def get_topk_action_by_bst(self, request, k):
        bst_delta = 10000
        myScores = []
        for server in self.servers:
            usable = server.usable(request)
            for numa in range(2):
                if usable[numa] == 1:
                    server.handle(request, numa)
                    if request['is_double'] == 1:
                        myScores.append(max(server.allo_rate(0,True), server.allo_rate(1,True)))
                        myScores.append(max(server.allo_rate(0,True), server.allo_rate(1,True)))
                        server.delete(request)
                        break
                    myScores.append(server.allo_rate(numa,True))
                    server.delete(request)
                else: 
                    myScores.append(bst_delta)
        if request['is_double'] == 1:
            k*=2
        avail_indexs = sorted(range(len(myScores)), key=lambda i: myScores[i])[:k]
        res = np.zeros(len(self.servers) * 2)
        for ind in avail_indexs:
            if myScores[ind] < bst_delta:
                res[ind] = 1
        return res

    def describe(self):
        '''
            return the server state
        '''
        des = []
        for server in self.servers:
            des.append(server.describe())
        return np.array(des)

    def check_usable(self, request):
        '''
            check servers is usable.
            return [[0,0], [0,1],...] for server's numa usable state
            1 is usable, 0 is not.
        '''
        usable_list = []
        for server in self.servers:
            usable_list.append(server.usable(request))
        res = np.array(usable_list).reshape(-1)
        return res

    def delete(self, request): 
        '''
            delete
        '''
        i = 0
        for server in self.servers:
            del_status = server.delete(request)
            if del_status != -1:
                return 2*i + del_status
            i += 1
        return -1

    def reset(self):
        '''
            reset the server state
        '''
        self.servers = [Server(self.cpu, self.mem) for i in range(self.N)]

    def cal_cpu_allocation_rate(self):
        '''
            return allocation rate of cpu
        '''
        allocation_rate=[]
        for server in self.servers:
            allocation_rate.append(1-server.allo_rate(0,True))
            allocation_rate.append(1-server.allo_rate(1,True))
        return np.mean(allocation_rate)

class SchedEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, N, cpu, mem, path, render_path=None, allow_release=False, rew_fn='iden',double_thr=1e10, topk=3):
        '''
            N is the number of servers, cpu and mem are the attribute of the server
            path is the csv path
        '''
        super(SchedEnv, self).__init__()
        self.N = N
        self.cpu = cpu
        self.mem = mem
        self.topk = topk
        self.cluster = Cluster(N, cpu, mem)
        self.requests = getData(path, double_thr)
        self.rew_fn = rew_fn
        self.max_delta = 5000
        self.t = 0
        self.dur = 0
        self.allow_release = allow_release

        self.render_list = []
        self.render_path = render_path
        self.isrender = False
        if render_path is not None:
            self.render_path = render_path
            self.isrender = True


    def get_property(self,):
        return self.N, self.cpu, self.mem

    def reset(self, step):
        '''
            return servers, request and usable list
        '''
        self.t = step
        self.start = step
        self.dur = 0
        self.cluster.reset()
        while self.requests[self.t]['type'] == 1:
            self.t += 1
        request = self.requests[self.t]
        if request['is_double'] == 1:
            request_info = [[[request['cpu']/2, request['mem']/2] for i in range(2)] for j in range(2)]
        else:
            request_info =[[[request['cpu'], request['mem']], [0,0]], [[0,0], [request['cpu'], request['mem']]]]
        state = self.cluster.describe()
        return state

    def termination(self):
        '''
            check if the env terminated
        '''
        is_term = False
        if self.t >= len(self.requests)-1:
            is_term = True
        else:
            request = self.requests[self.t]
            if np.array(self.cluster.check_usable(request)).sum() == 0:
                is_term = True
        if is_term and self.isrender:
            import pickle
            pickle.dump(self.render_list, open(self.render_path, "wb"))
        return is_term

    def handle_delete(self,):
        '''
            handle delete request until get to create request
        '''
        request = self.requests[self.t]
        while request['type'] == 1:
            if self.allow_release:
                del_status = self.cluster.delete(request)
                if self.isrender and del_status!=-1:
                    record_dict = {'server': self.cluster.describe(), 'request':self.requests[self.t], 'action': del_status}
                    self.render_list.append(record_dict)
            self.t += 1
            request = self.requests[self.t]
            if self.t >= len(self.requests):
                return False
        return True


    def reward(self):
        rew_fn = self.rew_fn
        if rew_fn == 'iden':
                return 1
        elif rew_fn == 'cpu_allo':
                return self.cluster.cal_cpu_allocation_rate()

    def step(self, action):
        '''
            env take action ,
            return state, reward and done
        '''
        action = self._step(action)
        if self.isrender:
            record_dict = {'server': self.cluster.describe(), 'request':self.requests[self.t], 'action': action}
            self.render_list.append(record_dict)
        self.t += 1
        self.dur += 1

        request = self.requests[self.t]
        if request['is_double'] == 1:
            request_info = [[[request['cpu']/2, request['mem']/2] for i in range(2)] for j in range(2)]
        else:
            request_info =[[[request['cpu'], request['mem']], [0,0]], [[0,0], [request['cpu'], request['mem']]]]
        self.handle_delete()

        state =  self.cluster.describe()
        reward = self.reward()
        done  = self.termination()

        avail = self.get_attr('avail')
        feat = self.get_attr('req')                  

        return action, state, reward, done, avail, feat, state


    def _step(self, action):
        request = self.requests[self.t]
        if action == -1:
            action = self.cluster.first_fit_action(request)
        elif action == -2:
            action = self.cluster.best_fit_action(request,True)
        elif action == -3:
            action = self.cluster.best_fit_action(request,False)
        
        self.cluster.handle(action, request)
        return action

    def get_attr(self, attr_name):
        request = self.requests[self.t]
        if attr_name == 'req':
            if request['is_double'] == 1:
                request_info = [[[request['cpu']/2, request['mem']/2] for i in range(2)] for j in range(2)]
            else:
                request_info =[[[request['cpu'], request['mem']], [0,0]], [[0,0], [request['cpu'], request['mem']]]]
            return request_info
        elif attr_name == 'avail':
            # return self.cluster.check_usable(request)
            return self.cluster.get_topk_action_by_bst(request, self.topk)
            
        elif attr_name == 'obs':
            return self.cluster.describe()
        elif attr_name == 'req_step':
            return self.t
        return None