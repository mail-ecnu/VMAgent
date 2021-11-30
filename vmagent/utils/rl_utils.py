import numpy as np

def linear_decay(epoch, x, y):
    min_v, max_v = y[0], y[-1]
    start, end = x[0], x[-1]

    if epoch == start:
        return min_v

    eps = min_v

    for i, x_i in enumerate(x):
        if epoch <= x_i:
            interval = (y[i] - y[i - 1]) / (x_i - x[i - 1])
            eps = interval * (epoch - x[i - 1]) + y[i - 1]
            break

    return eps


def time_format(t):
    hour = t//3600
    t = t % 3600 
    mi = t//60 
    t = t % 60 
    se = t 
    return hour, mi, se
