import numpy as np
import pandas as pd
import pickle

from uuid import uuid4

from .config import cpu_max, mem_max


DEFAULT_DATA_DQN_NAME = 'dqn'
DEFAULT_DATA_DQN = pickle.load(open('./data/dqn.p', 'rb'))
DEFAULT_DATA_DQN_LEN = len(DEFAULT_DATA_DQN)
DEFAULT_DATA_DQN_UUID = str(uuid4())

DEFAULT_DATA_FF_NAME = 'firstfit'
DEFAULT_DATA_FF = pickle.load(open('./data/firstfit.p', 'rb'))
DEFAULT_DATA_FF_LEN = len(DEFAULT_DATA_FF)
DEFAULT_DATA_FF_UUID = str(uuid4())


def get_server_num(df, n_intervals=0):
    if n_intervals < 0:
        n_intervals = 0
    return len(df['server'][n_intervals])


def format_data(raw):
    return pd.DataFrame(map(lambda x: dict(
        cpu=x['request']['cpu'],
        mem=x['request']['mem'],
        is_double=x['request']['is_double'],
        request_type=x['request']['type'],
        server=x['server'],
        action=x['action']
    ), raw))


def expand(df: pd.DataFrame, length):
    # if (needed := length - len(df)) > 0:
    needed = length - len(df)
    if needed > 0:
        return pd.concat([df, df.iloc[[-1] * needed]], ignore_index=True)
    return df


def total_usage_data_helper(df):
    max_intervals = len(df)
    total_usage = [(0, 0)]
    for n_intervals in range(max_intervals):
        server_num = get_server_num(df, n_intervals)
        total_cpu = np.sum(cpu_max - df['server'][n_intervals][..., 0]) / (cpu_max * 2 * server_num) * 100
        total_mem = np.sum(mem_max - df['server'][n_intervals][..., 1]) / (mem_max * 2 * server_num) * 100
        total_usage.append((total_cpu, total_mem))
    return pd.DataFrame(total_usage, columns=['cpu', 'mem'])


def score_data_helper(df):
    df = pd.DataFrame.from_dict(df)
    current_score = 0
    scores = [current_score]
    for req in df['request_type']:
        if req:
            current_score += 1
        else:
            current_score -= 1
        scores.append(current_score)
    return pd.DataFrame(scores, columns=['score'])


if __name__ == '__main__':
    test_df = format_data(pickle.load(open('../data/dqn.p', 'rb')))
    from pprint import pprint
    pprint(test_df)
