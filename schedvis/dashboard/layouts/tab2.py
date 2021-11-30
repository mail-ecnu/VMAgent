import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_daq as daq
import dash_html_components as html

from .slider import build_slider
from ..data import DEFAULT_DATA_DQN_NAME, DEFAULT_DATA_FF_NAME
from ..figures.tab2 import get_figure0, get_figure1, get_figure2, get_figure3, get_gauge_value, get_gauge_color


def build_tab(n_intervals, df0, df1, total_usage0, total_usage1, scores0, scores1, title):
    # cpu_color, mem_color = get_gauge_color(cpu), get_gauge_color(mem)
    return dbc.Container(
        fluid=True,
        children=[
            dbc.Container(
                fluid=True,
                children=[
                    dcc.Interval(
                        id='tab2-interval-component',
                        interval=500,
                        n_intervals=0,
                        max_intervals=-1,
                        disabled=True
                    ),
                    dcc.Store(id='tab2-interval-done', data=0),


                ]
            ),
            dbc.Container(
                fluid=True,
                children=dbc.CardDeck(build_slider('tab2-', True))
            ),

            html.Br(),

            dbc.Container(
                fluid=True,
                children=dbc.CardDeck([
                    dbc.Card(
                        children=dbc.CardBody(children=[
                            html.H4(title, id='tab2-card-title',
                                className='card-title'
                            ),
                            dcc.Graph(id='tab2-graph3', figure=get_figure3(n_intervals, df0, df1, scores0, scores1,
                                                                           DEFAULT_DATA_DQN_NAME, DEFAULT_DATA_FF_NAME))
                        ]),
                        className="col-md-4",
                    ),
                    dbc.Card(body=True, children=dcc.Graph(figure=get_figure1(n_intervals, df0),
                                                           id='tab2-graph1'), className="col-md-2"),
                    dbc.Card(body=True, children=dcc.Graph(
                        figure=get_figure2(n_intervals, total_usage0, total_usage1,
                                           DEFAULT_DATA_DQN_NAME, DEFAULT_DATA_FF_NAME),
                        id='tab2-graph2'), className="col-md-6"),
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
                                id='tab2-graph0',
                                figure=get_figure0(n_intervals, df0, df1, DEFAULT_DATA_DQN_NAME, DEFAULT_DATA_FF_NAME)
                            )
                        )
                    ],
                )
            )
        ]
    )
