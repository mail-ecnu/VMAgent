import multiprocessing
from collections import OrderedDict
from typing import Sequence

import gym
import numpy as np

from stable_baselines.common.vec_env.base_vec_env import VecEnv, CloudpickleWrapper


def _worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.var()
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == 'step':
                action, observation, reward, done = env.step(data)
                # if done:
                    # save final observation where user can get it, then reset
                    # observation = env.reset()
                remote.send((action, observation, reward, done))
            # elif cmd == 'seed':
            #     remote.send(env.seed(data))
            elif cmd == 'reset':
                observation = env.reset(data)
                remote.send(observation)
            elif cmd == 'get_attr':
                remote.send(env.get_attr(data))
            # elif cmd == 'set_attr':
            #     remote.send(setattr(env, data[0], data[1]))
            else:
                raise NotImplementedError("`{}` is not implemented in the worker".format(cmd))
        except EOFError:
            break



class SubprocVecEnv(VecEnv):
    """
    Creates a multiprocess vectorized wrapper for multiple environments, distributing each environment to its own
    process, allowing significant speed up when the environment is computationally complex.
    For performance reasons, if your environment is not IO bound, the number of environments should not exceed the
    number of logical cores on your CPU.
    .. warning::
        Only 'forkserver' and 'spawn' start methods are thread-safe,
        which is important when TensorFlow sessions or other non thread-safe
        libraries are used in the parent (see issue #217). However, compared to
        'fork' they incur a small start-up cost and have restrictions on
        global variables. With those methods, users must wrap the code in an
        ``if __name__ == "__main__":`` block.
        For more information, see the multiprocessing documentation.
    :param env_fns: ([callable]) A list of functions that will create the environments
        (each callable returns a `Gym.Env` instance when called).
    :param start_method: (str) method used to start the subprocesses.
           Must be one of the methods returned by multiprocessing.get_all_start_methods().
           Defaults to 'forkserver' on available platforms, and 'spawn' otherwise.
    """

    def __init__(self, env_fns, is_step_one_hot=False, start_method=None):
        self.waiting = False
        self.closed = False
        n_envs = len(env_fns)
        self.n_envs = n_envs
        self.alives = np.array([True for i in range(n_envs)])
        self.alives2 = np.array([True for i in range(n_envs)])
        self.is_step_one_hot = is_step_one_hot

        if start_method is None:
            # Fork is not a thread safe method (see issue #217)
            # but is more user friendly (does not require to wrap the code in
            # a `if __name__ == "__main__":`)
            forkserver_available = 'forkserver' in multiprocessing.get_all_start_methods()
            start_method = 'forkserver' if forkserver_available else 'spawn'
        ctx = multiprocessing.get_context(start_method)

        self.remotes, self.work_remotes = zip(*[ctx.Pipe(duplex=True) for _ in range(n_envs)])
        self.remotes = np.array(self.remotes)
        self.processes = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            args = (work_remote, remote, CloudpickleWrapper(env_fn))
            # daemon=True: if the main process crashes, we should not cause things to hang
            if self.is_step_one_hot:
                process = ctx.Process(target=_worker_step_oh, args=args, daemon=True)  # pytype:disable=attribute-error
            else:
                process = ctx.Process(target=_worker, args=args, daemon=True)  # pytype:disable=attribute-error
            process.start()
            self.processes.append(process)
            work_remote.close()

        VecEnv.__init__(self, len(env_fns), (1), (1))

    def step_async(self, actions):
        for remote, action in zip(self.remotes[self.alives], actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes[self.alives]]
        self.waiting = False
        
        if self.is_step_one_hot:
            act, obs, rews, dones, step_oh = zip(*results)
            self.alives2[self.alives] = ~np.array(dones)
            return np.stack(act), np.stack(obs), np.stack(rews), np.stack(dones), np.stack(step_oh)
        else:
            act, obs, rews, dones = zip(*results)
            self.alives2[self.alives] = ~np.array(dones)
            return np.stack(act), np.stack(obs), np.stack(rews), np.stack(dones)

    def get_alives(self):
        return self.alives

    def update_alives(self):
        self.alives = self.alives2.copy()


    def reset(self, steps):
        # print(steps)
        # print(type(steps))
        if type(steps) is int:
            for remote in self.remotes:
                remote.send(('reset', steps))
        else:
            for remote, step in zip(self.remotes, steps):
                remote.send(('reset', step))
        obs = [remote.recv() for remote in self.remotes]
        self.alives = np.array([True for i in range(self.n_envs)])
        self.alives2 = np.array([True for i in range(self.n_envs)])
        return np.stack(obs)

    def close(self):
        pass

    def seed(self, seed=None):
        pass

    def get_attr(self, attr_name):
        target_remotes = self.remotes[self.alives]
        if target_remotes.shape[0] == 0:
            import pdb; pdb.set_trace()
        for remote in target_remotes:
            remote.send(('get_attr', attr_name))
        return np.array([remote.recv() for remote in target_remotes])

    def set_attr(self, attr_name, value, indices=None):
        pass

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        pass



