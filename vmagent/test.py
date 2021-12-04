
import numpy as np
from schedgym.sched_env import SchedEnv
from schedgym.mySubproc_vec_env import SubprocVecEnv



DATA_PATH = 'vmagent/data/dataset.csv'

def make_env(N, cpu, mem, allow_release, double_thr=1e10):
    def _init():
        env = SchedEnv(N, cpu, mem, DATA_PATH, render_path=None,
                       allow_release=allow_release, double_thr=double_thr)
        return env
    return _init



if __name__ == "__main__":

    env = SchedEnv(5, 40, 90, DATA_PATH, render_path='../test.p',
                   allow_release=False, double_thr=32)
    MAX_STEP = 1e4
    env.reset(np.random.randint(0, MAX_STEP))
    done = env.termination()
    while not done:
        feat = env.get_attr('req')
        obs = env.get_attr('obs')
        # sample by first fit
        avail = env.get_attr('avail')
        action = np.random.choice(np.where(avail == 1)[0])
        action, next_obs, reward, done = env.step(action)

