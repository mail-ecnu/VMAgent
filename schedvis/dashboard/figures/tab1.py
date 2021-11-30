from typing import Tuple, Any

import numpy as np
import plotly.graph_objects as go

from plotly.subplots import make_subplots

from . import cpu_color, mem_color, \
    cpu_delta_color, mem_delta_color,\
    cpu_color_rgb, mem_color_rgb, \
    heatmap_scale
from ..config import cpu_max, mem_max, y_axis_max
from ..data import get_server_num
from ..utils import rgba


cpu_hover = 'Server %{x}<br>' \
            f'Total CPU: {cpu_max} Core<br>' \
            'Used: %{customdata[0]} Core<br>' \
            'Occupancy: %{customdata[1]:.2f}%' \
            '<extra></extra>'

mem_hover = 'Server %{x}<br>' \
            f'Total MEM: {mem_max} GB<br>' \
            'Used: %{customdata[0]} GB<br>' \
            'Occupancy: %{customdata[1]:.2f}%' \
            '<extra></extra>'


def __delta_color_marker_helper(delta: Tuple[Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # insert -> True
    # delete -> False
    return (
        np.array(list(cpu_delta_color[x] for x in delta[0] > 0)),
        np.array(list(mem_delta_color[x] for x in delta[1] > 0)),
        np.array(list(cpu_delta_color[x] for x in delta[2] < 0)),
        np.array(list(mem_delta_color[x] for x in delta[3] < 0)),
    )


def _get_figure0_small(server_num, n_intervals, df, show_delta):

    if n_intervals - show_delta < 0:
        trace_data = (np.zeros(server_num),) * 4
    else:
        trace_data = (
            cpu_max - df['server'][n_intervals - show_delta][..., 0, 0],
            mem_max - df['server'][n_intervals - show_delta][..., 0, 1],
            df['server'][n_intervals - show_delta][..., 1, 0] - cpu_max,
            df['server'][n_intervals - show_delta][..., 1, 1] - mem_max
        )

    rate = (
        trace_data[0] / cpu_max * 100,
        trace_data[1] / mem_max * 100,
        trace_data[2] / cpu_max * 100,
        trace_data[3] / mem_max * 100,
    )

    server_list = list(range(server_num))
    fig = go.Figure(layout=dict(yaxis=dict(
        range=[-y_axis_max, y_axis_max], autorange=False,
        tickmode='array',
        tickvals=list(range(-200, 200, 50)),
        ticktext=list(map(abs, range(-200, 200, 50))),
        zeroline=True, zerolinecolor='gray'
    )))


    fig.add_traces([
        go.Bar(
            x=np.append(server_list, server_list),
            y=np.append(trace_data[0], trace_data[2]),
            offsetgroup='C',
            customdata=np.array([np.append(trace_data[0], -trace_data[2]), np.append(rate[0], -rate[2])]).T,
            hovertemplate=cpu_hover,
            marker_color=cpu_color, # opacity=0.8,
            name='CPU'
        ),
        go.Bar(
            x=np.append(server_list, server_list),
            y=np.append(trace_data[1], trace_data[3]),
            offsetgroup='M',
            customdata=np.array([np.append(trace_data[1], -trace_data[3]), np.append(rate[1], -rate[3])]).T,
            hovertemplate=mem_hover,
            marker_color=mem_color, # opacity=0.8,
            name='MEM'
        ),
    ])

    if not show_delta:
        return fig

    next_trace_data = (
        cpu_max - df['server'][n_intervals][..., 0, 0],
        mem_max - df['server'][n_intervals][..., 0, 1],
        df['server'][n_intervals][..., 1, 0] - cpu_max,
        df['server'][n_intervals][..., 1, 1] - mem_max
    )

    delta_data = tuple(i - j for i, j in zip(next_trace_data, trace_data))
    # Delta
    delta_color = __delta_color_marker_helper(delta_data)

    fig.add_traces([
        go.Bar(
            x=np.append(server_list, server_list),
            y=np.append(delta_data[0], delta_data[2]),
            offsetgroup='C',
            base=np.append(trace_data[0], trace_data[2]),
            showlegend=False, name='CPU',
            hoverinfo='none',
            marker_color=np.append(delta_color[0], delta_color[2]),
            # marker_line_color=np.append(delta_color[0], delta_color[2]),
            # marker_line_width=1.5, # opacity=0.8,
        ),
        go.Bar(
            x=np.append(server_list, server_list),
            y=np.append(delta_data[1], delta_data[3]),
            offsetgroup='M',
            base=np.append(trace_data[1], trace_data[3]),
            showlegend=False, name='MEM',
            hoverinfo='none',
            marker_color=np.append(delta_color[1], delta_color[3]),
            # marker_line_color=np.append(delta_color[1], delta_color[2]),
            # marker_line_width=1.5, # opacity=0.8,
        ),
    ])
    return fig


def _get_figure0_big(server_num, n_intervals, df, show_delta=True):

    def __get_size(n):
        sz = {
            50: (10, 5),
            100: (10, 10),
            120: (12, 10),
            200: (20, 10)
        }
        return sz[n]

    w, h = __get_size(server_num)

    fig = make_subplots(1, 2, column_width=[2/3, 1/3])

    if n_intervals - show_delta < 0:
        hm_trace_data = np.zeros(w * h)
    else:
        __cpu_data = cpu_max - df['server'][n_intervals - show_delta][..., 0, 0] + \
                     cpu_max - df['server'][n_intervals - show_delta][..., 1, 0]
        __mem_data = mem_max - df['server'][n_intervals - show_delta][..., 0, 1] + \
                     mem_max - df['server'][n_intervals - show_delta][..., 1, 1]
        __cpu_data = __cpu_data / (cpu_max * 2) * 100
        __mem_data = __mem_data / (mem_max * 2) * 100
        hm_trace_data = np.maximum(__cpu_data, __mem_data)


    fig.add_trace(
        go.Heatmap(
            z=np.reshape(hm_trace_data, (w, h)).T,
            colorscale=heatmap_scale,
            zmin=0, zmax=100,
            colorbar_x=2/3 - 0.05,
            legendgroup='1',
            xgap=3, ygap=3
        ), row=1, col=1
    )

    fig.update_yaxes(
        range=[-y_axis_max, y_axis_max], autorange=False,
        tickmode='array',
        tickvals=list(range(-200, 200, 50)),
        ticktext=list(map(abs, range(-200, 200, 50))),
        zeroline=True, zerolinecolor='gray', row=1, col=2)

    request_server = df['action'][n_intervals] >> 1
    x, y = request_server // h, request_server % h
    fig.add_shape(dict(type='rect', x0=x - 0.5, x1=x + 0.5, y0=y - 0.5, y1=y + 0.5), row=1, col=1)

    if n_intervals - show_delta < 0:
        trace_data = (np.zeros(1),) * 4
    else:
        trace_data = (
            cpu_max - df['server'][n_intervals - show_delta][request_server: request_server + 1, 0, 0],
            mem_max - df['server'][n_intervals - show_delta][request_server: request_server + 1, 0, 1],
            df['server'][n_intervals - show_delta][request_server: request_server + 1, 1, 0] - cpu_max,
            df['server'][n_intervals - show_delta][request_server: request_server + 1, 1, 1] - mem_max
        )

    rate = (
        trace_data[0] / cpu_max * 100,
        trace_data[1] / mem_max * 100,
        trace_data[2] / cpu_max * 100,
        trace_data[3] / mem_max * 100,
    )

    fig.add_traces([
        go.Bar(
            x=np.zeros(2),
            y=np.append(trace_data[0], trace_data[2]),
            offsetgroup='C',
            customdata=np.array([np.append(trace_data[0], -trace_data[2]), np.append(rate[0], -rate[2])]).T,
            hovertemplate=cpu_hover,
            marker_color=cpu_color, # opacity=0.8,
            name='CPU',
            legendgroup='2'
        ),
        go.Bar(
            x=np.zeros(2),
            y=np.append(trace_data[1], trace_data[3]),
            offsetgroup='M',
            customdata=np.array([np.append(trace_data[1], -trace_data[3]), np.append(rate[1], -rate[3])]).T,
            hovertemplate=mem_hover,
            marker_color=mem_color, # opacity=0.8,
            name='MEM',
            legendgroup='2'
        ),
    ], rows=1, cols=2)

    if not show_delta:
        return fig

    next_trace_data = (
        cpu_max - df['server'][n_intervals][request_server: request_server + 1, 0, 0],
        mem_max - df['server'][n_intervals][request_server: request_server + 1, 0, 1],
        df['server'][n_intervals][request_server: request_server + 1, 1, 0] - cpu_max,
        df['server'][n_intervals][request_server: request_server + 1, 1, 1] - mem_max
    )

    delta_data = tuple(i - j for i, j in zip(next_trace_data, trace_data))
    # Delta
    delta_color = __delta_color_marker_helper(delta_data)

    fig.add_traces([
        go.Bar(
            x=np.zeros(2),
            y=np.append(delta_data[0], delta_data[2]),
            offsetgroup='C',
            base=np.append(trace_data[0], trace_data[2]),
            showlegend=False, name='CPU',
            hoverinfo='none',
            marker_color=np.append(delta_color[0], delta_color[2]),
            # marker_line_color=np.append(delta_color[0], delta_color[2]),
            # marker_line_width=1.5,
            # opacity=0.8,
            legendgroup='2'
        ),
        go.Bar(
            x=np.zeros(2),
            y=np.append(delta_data[1], delta_data[3]),
            offsetgroup='M',
            base=np.append(trace_data[1], trace_data[3]),
            showlegend=False, name='MEM',
            hoverinfo='none',
            marker_color=np.append(delta_color[1], delta_color[3]),
            # marker_line_color=np.append(delta_color[1], delta_color[2]),
            # marker_line_width=1.5,
            # opacity=0.8,
            legendgroup='2'
        ),
    ], rows=1, cols=2)
    return fig


def get_figure0(n_intervals, df, show_delta=True, threshold=50):
    server_num = get_server_num(df, n_intervals - show_delta)
    callback_func = (_get_figure0_small, _get_figure0_big)[server_num >= threshold]
    return callback_func(server_num, n_intervals, df, show_delta)


def get_figure1(n_intervals, df):
    cpu = df['cpu'][n_intervals]
    mem = df['mem'][n_intervals]
    is_double = df['is_double'][n_intervals]
    request_type = df['request_type'][n_intervals]

    if is_double:
        data = np.array([[cpu / 2], [mem / 2], [-cpu / 2], [-mem / 2]])
    else:
        data = np.array([[cpu], [mem], [0], [0]])
    yaxis_range = [-y_axis_max, y_axis_max] if np.any(np.abs(data) > 50) else [-50, 50]
    fig = go.Figure(layout=dict(barmode='relative', yaxis=dict(
        range=yaxis_range, autorange=False,
        tickmode='array',
        tickvals=list(range(-100, 100,  25)),
        ticktext=list(map(abs, range(-100, 100, 25)))
    )))
    fig.add_traces([
        go.Bar(
            x=['CPU', 'CPU'], y=np.append(data[0], data[2]),
            offsetgroup='C', marker_color=cpu_delta_color[request_type],
            showlegend=False, name='CPU'
        ),
        go.Bar(
            x=['MEM', 'MEM'], y=np.append(data[1], data[3]),
            offsetgroup='M', marker_color=mem_delta_color[request_type],
            showlegend=False, name='MEM'
        ),
    ])
    return fig


def get_figure2(n_intervals, df):
    wd_size = 300
    start = max(0, n_intervals - wd_size)
    fig = go.Figure(layout=dict(xaxis=dict(range=[start, max(n_intervals, wd_size)]),
                                yaxis=dict(rangemode='nonnegative')))
    fig.add_traces([
        go.Scatter(
            x=np.arange(n_intervals + 1), y=df['cpu'][:n_intervals + 1], name='CPU', fill='tozeroy',
            fillcolor=rgba(*cpu_color_rgb, 0.25), line=dict(color=cpu_color),
            mode='lines'
        ),
        go.Scatter(
            x=np.arange(n_intervals + 1), y=df['mem'][:n_intervals + 1], name='MEM', fill='tozeroy',
            fillcolor=rgba(*mem_color_rgb, 0.5), line=dict(color=mem_color),
            mode='lines'
        ),
    ])
    return fig


def get_gauge_value(n_intervals, df):
    if n_intervals == 0:
        return 0, 0
    server_num = get_server_num(df, n_intervals - 1)
    total_cpu = np.sum(cpu_max - df['server'][n_intervals - 1][..., 0]) / (cpu_max * 2 * server_num) * 100
    total_mem = np.sum(mem_max - df['server'][n_intervals - 1][..., 1]) / (mem_max * 2 * server_num) * 100
    total_cpu = int(total_cpu * 10) / 10
    total_mem = int(total_mem * 10) / 10
    return total_cpu, total_mem


def get_gauge_color(value):
    if value < 50:
        return 'var(--info)'
    elif value < 80:
        return 'var(--warning)'
    return 'var(--danger)'
