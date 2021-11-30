from typing import List, Tuple, Any

import numpy as np
import plotly.graph_objects as go

from plotly.subplots import make_subplots

from . import cpu_color, mem_color, \
    cpu_delta_color, mem_delta_color, \
    cpu_color_rgb, mem_color_rgb, socre_color, heatmap_scale
from .tab1 import get_figure1 as get_tab1_figure1
from .tab1 import __delta_color_marker_helper as __tab1_delta_color_maker_helper
from ..config import cpu_max, mem_max, y_axis_max
from ..data import get_server_num
from ..utils import rgb, rgba


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


def _delta_color_marker_helper(delta: Tuple[Any], delta_color: Tuple[Any, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # insert -> True
    # delete -> False
    return (
        np.array(list(delta_color[x] for x in delta[0] > 0)),
        np.array(list(delta_color[x] for x in delta[1] < 0)),
        np.array(list(delta_color[x] for x in delta[2] > 0)),
        np.array(list(delta_color[x] for x in delta[3] < 0)),
    )


def _get_figure0_small(server_num, n_intervals, df0, df1, name0, name1, show_delta):

    # CPU
    if n_intervals - show_delta < 0:
        cpu_trace_data = (np.zeros(server_num),) * 4
        mem_trace_data = (np.zeros(server_num),) * 4
    else:
        cpu_trace_data = (
            cpu_max - df0['server'][n_intervals - show_delta][..., 0, 0],
            df0['server'][n_intervals - show_delta][..., 1, 0] - cpu_max,
            cpu_max - df1['server'][n_intervals - show_delta][..., 0, 0],
            df1['server'][n_intervals - show_delta][..., 1, 0] - cpu_max,
        )
        mem_trace_data = (
            mem_max - df0['server'][n_intervals - show_delta][..., 0, 1],
            df0['server'][n_intervals - show_delta][..., 1, 1] - mem_max,
            mem_max - df1['server'][n_intervals - show_delta][..., 0, 1],
            df1['server'][n_intervals - show_delta][..., 1, 1] - mem_max,
        )

    cpu_rate = (
        cpu_trace_data[0] / cpu_max * 100,
        cpu_trace_data[1] / cpu_max * 100,
        cpu_trace_data[2] / cpu_max * 100,
        cpu_trace_data[3] / cpu_max * 100,
    )
    mem_rate = (
        mem_trace_data[0] / mem_max * 100,
        mem_trace_data[1] / mem_max * 100,
        mem_trace_data[2] / mem_max * 100,
        mem_trace_data[3] / mem_max * 100,
    )

    server_list = list(range(server_num))
    fig = make_subplots(rows=1, cols=2)
    fig.update_yaxes(
        range=[-y_axis_max, y_axis_max], autorange=False,
        tickmode='array',
        tickvals=list(range(-200, 200, 50)),
        ticktext=list(map(abs, range(-200, 200, 50))),
        zeroline=True, zerolinecolor='gray'
    )

    fig.add_traces([
        go.Bar(
            x=np.append(server_list, server_list),
            y=np.append(cpu_trace_data[0], cpu_trace_data[1]),
            offsetgroup='C1',
            customdata=np.array([np.append(cpu_trace_data[0], -cpu_trace_data[1]),
                                 np.append(cpu_rate[0], -cpu_rate[1])]).T,
            hovertemplate=cpu_hover,
            marker_color=cpu_color, # opacity=0.8,
            name=f'CPU - {name0}'
        ),
        go.Bar(
            x=np.append(server_list, server_list),
            y=np.append(cpu_trace_data[2], cpu_trace_data[3]),
            offsetgroup='C2',
            customdata=np.array([np.append(cpu_trace_data[2], -cpu_trace_data[3]),
                                 np.append(cpu_rate[2], -cpu_rate[3])]).T,
            hovertemplate=cpu_hover,
            marker_color=cpu_color, # opacity=0.8,
            marker_pattern_fillmode='overlay',
            marker_pattern_shape='/',
            name=f'CPU - {name1}'
        ),
    ], rows=1, cols=1)

    fig.add_traces([
        go.Bar(
            x=np.append(server_list, server_list),
            y=np.append(mem_trace_data[0], mem_trace_data[1]),
            offsetgroup='M1',
            customdata=np.array([np.append(mem_trace_data[0], -mem_trace_data[1]),
                                 np.append(mem_rate[0], -mem_rate[1])]).T,
            hovertemplate=mem_hover,
            marker_color=mem_color, # opacity=0.8,
            name=f'MEM - {name0}'
        ),
        go.Bar(
            x=np.append(server_list, server_list),
            y=np.append(mem_trace_data[2], mem_trace_data[3]),
            offsetgroup='M2',
            customdata=np.array([np.append(mem_trace_data[2], -mem_trace_data[3]),
                                 np.append(mem_rate[2], -mem_rate[3])]).T,
            hovertemplate=mem_hover,
            marker_color=mem_color, # opacity=0.8,
            marker_pattern_fillmode='overlay',
            marker_pattern_shape='/',
            name=f'MEM - {name1}'
        ),
    ], rows=1, cols=2)

    if not show_delta:
        return fig

    next_cpu_trace_data = (
        cpu_max - df0['server'][n_intervals][..., 0, 0],
        df0['server'][n_intervals][..., 1, 0] - cpu_max,
        cpu_max - df1['server'][n_intervals][..., 0, 0],
        df1['server'][n_intervals][..., 1, 0] - cpu_max,
    )
    next_mem_trace_data = (
        mem_max - df0['server'][n_intervals][..., 0, 1],
        df0['server'][n_intervals][..., 1, 1] - mem_max,
        mem_max - df1['server'][n_intervals][..., 0, 1],
        df1['server'][n_intervals][..., 1, 1] - mem_max,
    )

    cpu_delta_data = tuple(i - j for i, j in zip(next_cpu_trace_data, cpu_trace_data))
    mem_delta_data = tuple(i - j for i, j in zip(next_mem_trace_data, mem_trace_data))

    # Delta
    cpu_delta_colors = _delta_color_marker_helper(cpu_delta_data, cpu_delta_color)
    mem_delta_colors = _delta_color_marker_helper(mem_delta_data, mem_delta_color)

    fig.add_traces([
        go.Bar(
            x=np.append(server_list, server_list),
            y=np.append(cpu_delta_data[0], cpu_delta_data[1]),
            offsetgroup='C1',
            base=np.append(cpu_trace_data[0], cpu_trace_data[1]),
            showlegend=False, name=f'CPU - {name0}',
            hoverinfo='none',
            marker_color=np.append(cpu_delta_colors[0], cpu_delta_colors[1]),
        ),
        go.Bar(
            x=np.append(server_list, server_list),
            y=np.append(cpu_delta_data[2], cpu_delta_data[3]),
            offsetgroup='C2',
            base=np.append(cpu_trace_data[2], cpu_trace_data[3]),
            showlegend=False, name=f'CPU - {name1}',
            hoverinfo='none',
            marker_color=np.append(cpu_delta_colors[2], cpu_delta_colors[3]),
            marker_pattern_fillmode='overlay',
            marker_pattern_shape='/',
        ),
    ], rows=1, cols=1)

    fig.add_traces([
        go.Bar(
            x=np.append(server_list, server_list),
            y=np.append(mem_delta_data[0], mem_delta_data[1]),
            offsetgroup='M1',
            base=np.append(mem_trace_data[0], mem_trace_data[1]),
            showlegend=False, name=f'MEM - {name0}',
            hoverinfo='none',
            marker_color=np.append(mem_delta_colors[0], mem_delta_colors[1]),
        ),
        go.Bar(
            x=np.append(server_list, server_list),
            y=np.append(mem_delta_data[2], mem_delta_data[3]),
            offsetgroup='M2',
            base=np.append(mem_trace_data[2], mem_trace_data[3]),
            showlegend=False, name=f'MEM - {name1}',
            hoverinfo='none',
            marker_color=np.append(mem_delta_colors[2], mem_delta_colors[3]),
            marker_pattern_fillmode='overlay',
            marker_pattern_shape='/',
        ),
    ], rows=1, cols=2)

    return fig


def _get_figure0_big(server_num, n_intervals, df0, df1, name0, name1, show_delta=True):

    def __get_size(n):
        sz = {
            50: (10, 5),
            100: (10, 10),
            120: (12, 10),
            200: (20, 10)
        }
        return sz[n]

    w, h = __get_size(server_num)

    fig = make_subplots(rows=2, cols=3, column_width=[1/2, 1/4, 1/4],
                        specs=[[{}, {"rowspan": 2}, {"rowspan": 2}], [{}, None, None]],
                        subplot_titles=[name0, name0, name1, name1])

    if n_intervals - show_delta < 0:
        hm_trace_data0 = np.zeros(server_num)
        hm_trace_data1 = np.zeros(server_num)
    else:
        __cpu_data0 = cpu_max - df0['server'][n_intervals - show_delta][..., 0, 0] + \
                      cpu_max - df0['server'][n_intervals - show_delta][..., 1, 0]
        __mem_data0 = mem_max - df0['server'][n_intervals - show_delta][..., 0, 1] + \
                      mem_max - df0['server'][n_intervals - show_delta][..., 1, 1]
        __cpu_data0 = __cpu_data0 / (cpu_max * 2) * 100
        __mem_data0 = __mem_data0 / (mem_max * 2) * 100
        hm_trace_data0 = np.maximum(__cpu_data0, __mem_data0)

        __cpu_data1 = cpu_max - df1['server'][n_intervals - show_delta][..., 0, 0] + \
                      cpu_max - df1['server'][n_intervals - show_delta][..., 1, 0]
        __mem_data1 = mem_max - df1['server'][n_intervals - show_delta][..., 0, 1] + \
                      mem_max - df1['server'][n_intervals - show_delta][..., 1, 1]
        __cpu_data1 = __cpu_data1 / (cpu_max * 2) * 100
        __mem_data1 = __mem_data1 / (mem_max * 2) * 100
        hm_trace_data1 = np.maximum(__cpu_data1, __mem_data1)

    fig.add_trace(
        go.Heatmap(
            z=np.reshape(hm_trace_data0, (w, h)).T,
            colorscale=heatmap_scale,
            zmin=0, zmax=100,
            colorbar_x=1/2 - 0.05,
            legendgroup='1',
            xgap=3, ygap=3,
            name=name0
        ), row=1, col=1
    )

    fig.add_trace(
        go.Heatmap(
            z=np.reshape(hm_trace_data1, (w, h)).T,
            colorscale=heatmap_scale,
            zmin=0, zmax=100,
            colorbar_x=1/2 - 0.05,
            legendgroup='1',
            xgap=3, ygap=3,
            name=name1
        ), row=2, col=1
    )

    fig.update_yaxes(
        range=[-160, 160], autorange=False,
        tickmode='array',
        tickvals=list(range(-200, 200, 50)),
        ticktext=list(map(abs, range(-200, 200, 50))),
        zeroline=True, zerolinecolor='gray', row=1, col=2)

    fig.update_yaxes(
        range=[-160, 160], autorange=False,
        tickmode='array',
        tickvals=list(range(-200, 200, 50)),
        ticktext=list(map(abs, range(-200, 200, 50))),
        zeroline=True, zerolinecolor='gray', row=1, col=3)

    request_server0 = df0['action'][n_intervals] >> 1
    request_server1 = df1['action'][n_intervals] >> 1
    x0, y0 = request_server0 // h, request_server0 % h
    x1, y1 = request_server1 // h, request_server1 % h
    fig.add_shape(dict(type='rect', x0=x0 - 0.5, x1=x0 + 0.5, y0=y0 - 0.5, y1=y0 + 0.5), row=1, col=1)
    fig.add_shape(dict(type='rect', x0=x1 - 0.5, x1=x1 + 0.5, y0=y1 - 0.5, y1=y1 + 0.5), row=2, col=1)

    if n_intervals - show_delta < 0:
        trace_data0 = (np.zeros(1),) * 4
        trace_data1 = (np.zeros(1),) * 4
    else:
        trace_data0 = (
            cpu_max - df0['server'][n_intervals - show_delta][request_server0: request_server0 + 1, 0, 0],
            mem_max - df0['server'][n_intervals - show_delta][request_server0: request_server0 + 1, 0, 1],
            df0['server'][n_intervals - show_delta][request_server0: request_server0 + 1, 1, 0] - cpu_max,
            df0['server'][n_intervals - show_delta][request_server0: request_server0 + 1, 1, 1] - mem_max
        )
        trace_data1 = (
            cpu_max - df1['server'][n_intervals - show_delta][request_server1: request_server1 + 1, 0, 0],
            mem_max - df1['server'][n_intervals - show_delta][request_server1: request_server1 + 1, 0, 1],
            df1['server'][n_intervals - show_delta][request_server1: request_server1 + 1, 1, 0] - cpu_max,
            df1['server'][n_intervals - show_delta][request_server1: request_server1 + 1, 1, 1] - mem_max
        )

    rate0 = (
        trace_data0[0] / cpu_max * 100,
        trace_data0[1] / mem_max * 100,
        trace_data0[2] / cpu_max * 100,
        trace_data0[3] / mem_max * 100,
    )
    rate1 = (
        trace_data1[0] / cpu_max * 100,
        trace_data1[1] / mem_max * 100,
        trace_data1[2] / cpu_max * 100,
        trace_data1[3] / mem_max * 100,
    )

    fig.add_traces([
        go.Bar(
            x=np.zeros(2),
            y=np.append(trace_data0[0], trace_data0[2]),
            offsetgroup='C',
            customdata=np.array([np.append(trace_data0[0], -trace_data0[2]), np.append(rate0[0], -rate0[2])]).T,
            hovertemplate=cpu_hover,
            marker_color=cpu_color, # opacity=0.8,
            name=f'CPU - {name0}',
            legendgroup=name0
        ),
        go.Bar(
            x=np.zeros(2),
            y=np.append(trace_data0[1], trace_data0[3]),
            offsetgroup='M',
            customdata=np.array([np.append(trace_data0[1], -trace_data0[3]), np.append(rate0[1], -rate0[3])]).T,
            hovertemplate=mem_hover,
            marker_color=mem_color, # opacity=0.8,
            name=f'MEM - {name0}',
            legendgroup=name0
        ),
    ], rows=1, cols=2)


    fig.add_traces([
        go.Bar(
            x=np.zeros(2),
            y=np.append(trace_data1[0], trace_data1[2]),
            offsetgroup='C',
            customdata=np.array([np.append(trace_data1[0], -trace_data1[2]), np.append(rate1[0], -rate1[2])]).T,
            hovertemplate=cpu_hover,
            marker_color=cpu_color, # opacity=0.8,
            name=f'CPU - {name1}',
            legendgroup=name1
        ),
        go.Bar(
            x=np.zeros(2),
            y=np.append(trace_data1[1], trace_data1[3]),
            offsetgroup='M',
            customdata=np.array([np.append(trace_data1[1], -trace_data1[3]), np.append(rate1[1], -rate1[3])]).T,
            hovertemplate=mem_hover,
            marker_color=mem_color, # opacity=0.8,
            name=f'MEM - {name1}',
            legendgroup=name1
        ),
    ], rows=1, cols=3)

    if not show_delta:
        return fig

    next_trace_data0 = (
        cpu_max - df0['server'][n_intervals][request_server0: request_server0 + 1, 0, 0],
        mem_max - df0['server'][n_intervals][request_server0: request_server0 + 1, 0, 1],
        df0['server'][n_intervals][request_server0: request_server0 + 1, 1, 0] - cpu_max,
        df0['server'][n_intervals][request_server0: request_server0 + 1, 1, 1] - mem_max
    )
    next_trace_data1 = (
        cpu_max - df1['server'][n_intervals][request_server1: request_server1 + 1, 0, 0],
        mem_max - df1['server'][n_intervals][request_server1: request_server1 + 1, 0, 1],
        df1['server'][n_intervals][request_server1: request_server1 + 1, 1, 0] - cpu_max,
        df1['server'][n_intervals][request_server1: request_server1 + 1, 1, 1] - mem_max
    )

    delta_data0 = tuple(i - j for i, j in zip(next_trace_data0, trace_data0))
    delta_data1 = tuple(i - j for i, j in zip(next_trace_data1, trace_data1))
    # Delta
    delta_color0 = __tab1_delta_color_maker_helper(delta_data0)
    delta_color1 = __tab1_delta_color_maker_helper(delta_data1)

    fig.add_traces([
        go.Bar(
            x=np.zeros(2),
            y=np.append(delta_data0[0], delta_data0[2]),
            offsetgroup='C',
            base=np.append(trace_data0[0], trace_data0[2]),
            showlegend=False, name=f'CPU - {name0}',
            hoverinfo='none',
            marker_color=np.append(delta_color0[0], delta_color0[2]),
            legendgroup=name0
        ),
        go.Bar(
            x=np.zeros(2),
            y=np.append(delta_data0[1], delta_data0[3]),
            offsetgroup='M',
            base=np.append(trace_data0[1], trace_data0[3]),
            showlegend=False, name=f'MEM - {name0}',
            hoverinfo='none',
            marker_color=np.append(delta_color0[1], delta_color0[3]),
            legendgroup=name0
        ),
    ], rows=1, cols=2)


    fig.add_traces([
        go.Bar(
            x=np.zeros(2),
            y=np.append(delta_data1[0], delta_data1[2]),
            offsetgroup='C',
            base=np.append(trace_data1[0], trace_data1[2]),
            showlegend=False, name=f'CPU - {name1}',
            hoverinfo='none',
            marker_color=np.append(delta_color1[0], delta_color1[2]),
            legendgroup=name1
        ),
        go.Bar(
            x=np.zeros(2),
            y=np.append(delta_data1[1], delta_data1[3]),
            offsetgroup='M',
            base=np.append(trace_data1[1], trace_data1[3]),
            showlegend=False, name=f'MEM - {name1}',
            hoverinfo='none',
            marker_color=np.append(delta_color1[1], delta_color1[3]),
            legendgroup=name1
        ),
    ], rows=1, cols=3)
    return fig


def get_figure0(n_intervals, df0, df1, name0, name1, show_delta=True, threshold=50):
    server_num = get_server_num(df0, n_intervals - show_delta)
    callback_func = (_get_figure0_small, _get_figure0_big)[server_num >= threshold]
    return callback_func(server_num, n_intervals, df0, df1, name0, name1, show_delta)


def get_figure1(*args, **kwargs):
    return get_tab1_figure1(*args, **kwargs)


def get_figure2(n_intervals, df0, df1, name0, name1):
    wd_size = 300
    start = max(0, n_intervals - wd_size)
    fig = go.Figure(layout=dict(xaxis=dict(range=[start, max(n_intervals, wd_size)], exponentformat='none'),
                                yaxis=dict(rangemode='nonnegative')))
    fig.add_traces([
        go.Scatter(
            x=np.arange(n_intervals + 1), y=df0['cpu'][:n_intervals + 1], name=f'CPU - {name0}', fill='tozeroy',
            fillcolor=rgba(*cpu_color_rgb, 0.2), line=dict(color=cpu_color),
            mode='lines'
        ),
        go.Scatter(
            x=np.arange(n_intervals + 1), y=df1['cpu'][:n_intervals + 1], name=f'CPU - {name1}', fill='tozeroy',
            fillcolor=rgba(*cpu_color_rgb, 0.2), line=dict(color=cpu_color, dash='dash'),
            mode='lines'
        ),
        go.Scatter(
            x=np.arange(n_intervals + 1), y=df0['mem'][:n_intervals + 1], name=f'MEM - {name0}', fill='tozeroy',
            fillcolor=rgba(*mem_color_rgb, 0.4), line=dict(color=mem_color),
            mode='lines'
        ),
        go.Scatter(
            x=np.arange(n_intervals + 1), y=df1['mem'][:n_intervals + 1], name=f'MEM - {name1}', fill='tozeroy',
            fillcolor=rgba(*mem_color_rgb, 0.4), line=dict(color=mem_color, dash='dash'),
            mode='lines'
        ),
    ])
    return fig


def get_figure3(n_intervals, df0, df1, scores0, scores1, name0, name1):
    cpu0, mem0 = get_gauge_value(n_intervals, df0)
    cpu1, mem1 = get_gauge_value(n_intervals, df1)
    score0 = scores0['score'][n_intervals]
    score1 = scores1['score'][n_intervals]

    fig = make_subplots(rows=2, cols=2, specs=[[{"colspan": 2}, None], [{}, {}]], shared_yaxes=True)
    fig.update_yaxes(range=[0, (max(score0, score1) * 1.2 + 101) // 100 * 100], row=1, col=1)
    fig.update_yaxes(range=[0, 120], row=2)

    template = '%{y:.2f}%'

    fig.add_traces([
        go.Bar(x=['score'], y=[score0],
               marker_color=socre_color,
               # opacity=0.8,
               name=f'score - {name0}',
               texttemplate='%{y}',
               textposition='outside',
               textfont=dict(size=20)),
        go.Bar(x=['score'], y=[score1],
               marker_color=socre_color,
               # opacity=0.8,
               name=f'score - {name1}',
               marker_pattern_fillmode='overlay',
               marker_pattern_shape='/',
               texttemplate='%{y}',
               textposition='outside',
               textfont=dict(size=20))
    ], rows=1, cols=1)
    fig.add_traces([
        go.Bar(x=['CPU'], y=[cpu0], name=f'CPU - {name0}',
               marker_color=cpu_color,
               # opacity=0.8,
               texttemplate=template,
               textposition='outside',
               textfont=dict(color='black')),
        go.Bar(x=['CPU'], y=[cpu1], name=f'CPU - {name1}',
               marker_color=cpu_color,
               # opacity=0.8,
               marker_pattern_fillmode='overlay',
               marker_pattern_shape='/',
               texttemplate=template,
               textposition='outside',
               textfont=dict(color='black'))
    ], rows=2, cols=1)
    fig.add_traces([
        go.Bar(x=['MEM'], y=[mem0], name=f'MEM - {name0}',
               marker_color=mem_color,
               # opacity=0.8,
               texttemplate=template,
               textposition='outside',
               textfont=dict(color='black')),

        go.Bar(x=['MEM'], y=[mem1], name=f'MEM - {name1}',
               marker_color=mem_color,
               # opacity=0.8,
               marker_pattern_fillmode='overlay',
               marker_pattern_shape='/',
               texttemplate=template,
               textposition='outside',
               textfont=dict(color='black'))
    ], rows=2, cols=2)
    return fig


def get_gauge_value(n_intervals, df):
    if n_intervals == 0:
        return 0, 0

    server_num = get_server_num(df, n_intervals - 1)
    total_cpu = np.sum(cpu_max - df['server'][n_intervals - 1][..., 0]) / (cpu_max * 2 * server_num) * 100
    total_mem = np.sum(mem_max - df['server'][n_intervals - 1][..., 1]) / (mem_max * 2 * server_num) * 100
    return total_cpu, total_mem


def get_gauge_color(value):
    if value < 30:
        return '#A4D3EE'
    elif value < 50:
        return '#8DB6CD'
    elif value < 80:
        return '#FFD700'
    return '#EE9A00'
