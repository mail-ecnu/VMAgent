import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_daq as daq
import dash_html_components as html

from .slider import build_slider
from ..data import DEFAULT_DATA_DQN_NAME
from ..figures.tab1 import get_figure0, get_figure1, get_figure2, get_gauge_value, get_gauge_color


def build_tab(n_intervals, df, total_usage, scores, title):
    cpu, mem = get_gauge_value(n_intervals, df)
    cpu_color, mem_color = get_gauge_color(cpu), get_gauge_color(mem)
    score = scores['score'][n_intervals]
    return dbc.Container(
        fluid=True,
        children=[
            dbc.Container(
                fluid=True,
                children=[
                    dcc.Interval(
                        id='tab1-interval-component',
                        interval=500,
                        n_intervals=0,
                        max_intervals=-1,
                        disabled=True
                    ),
                    dcc.Store(id='tab1-interval-done', data=0),
                ]
            ),
            dbc.Container(
                fluid=True,
                children=dbc.CardDeck(build_slider('tab1-'))
            ),

            html.Br(),

            dbc.Container(
                fluid=True,
                children=dbc.CardDeck([
                    dbc.Card(
                        children=dbc.CardBody(children=[
                            html.H4(children=title, id='tab1-card-title', className='card-title'),
                            html.Div(children=[
                                daq.Gauge(
                                    id='tab1-gauge-cpu', showCurrentValue=True, value=0, color=cpu_color,
                                    units="%",
                                    label='CPU', max=100, min=0
                                ),
                                daq.LEDDisplay(
                                    id='tab1-led',
                                    label="Score",
                                    value=score,
                                    color='var(--primary)',
                                    style=dict(width='10%')
                                ),
                                daq.Gauge(
                                    id='tab1-gauge-mem', showCurrentValue=True, value=0, color=mem_color,
                                    units="%",
                                    label='MEM', max=100, min=0
                                )
                            ], style=dict(display='flex', justifyContent='center', alignItems='flex-end'))
                        ]),
                        className="col-md-4",
                    ),
                    dbc.Card(body=True, children=dcc.Graph(figure=get_figure1(0, df),
                                                           id='tab1-graph1'), className="col-md-2"),
                    dbc.Card(body=True, children=dcc.Graph(figure=get_figure2(0, total_usage),
                                                           id='tab1-graph2'), className="col-md-6"),
                ]),
            ),
            html.Br(),
            dbc.Container(
                fluid=True,
                children=dbc.CardDeck(
                    children=[
                        dbc.Card(
                            body=True,
                            children=dcc.Graph(
                                id='tab1-graph0', figure=get_figure0(0, df),
                            )
                        )
                    ]
                )
            )
        ]
    )
