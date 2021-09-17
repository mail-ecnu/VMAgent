import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html

from . import callbacks             # MUST import all callbacks
from .data import DEFAULT_DATA_DQN_LEN, DEFAULT_DATA_FF_LEN,\
    DEFAULT_DATA_DQN_UUID, DEFAULT_DATA_FF_UUID,\
    DEFAULT_DATA_DQN_NAME, DEFAULT_DATA_FF_NAME
from .server import app, server     # MUST import server for wsgi


def build_tabs():
    return html.Div(
        id='tabs',
        className='tabs',
        children=[
            dbc.Tabs(
                id='app-tabs',
                children=[
                    dbc.Tab(
                        id='tab-1',
                        label='single',
                        tab_id='tab1',
                    ),
                    dbc.Tab(
                        id='tab-2',
                        label='double',
                        tab_id='tab2',
                    )
                ]
            )
        ]
    )


app.layout = dbc.Container(
    id='big-app-container',
    className='dbc_light',
    fluid=True,
    children=[
        html.Br(),
        dbc.Jumbotron(
            id='app-container',
            # fluid=True,
            children=[
                html.H1(children='VMAgent', className="display-3", style=dict(textTransform='none')),
                html.Hr(),
                build_tabs(),
                html.Br(),
                dbc.Container(
                    fluid=True,
                    children=[
                        html.Div(id='app-content'),
                    ]
                ),
                # Data Upload
                dcc.Store(id='tab1-max-intervals', data=DEFAULT_DATA_DQN_LEN),
                dcc.Store(id='tab1-data-frame', data=DEFAULT_DATA_DQN_UUID),
                dcc.Store(id='tab1-card-title-data', data=DEFAULT_DATA_DQN_NAME),
                # Data Upload
                dcc.Store(id='tab2-max-intervals', data=max(DEFAULT_DATA_DQN_LEN, DEFAULT_DATA_FF_LEN)),
                dcc.Store(id='tab2-data-frame', data=[DEFAULT_DATA_DQN_UUID, DEFAULT_DATA_FF_UUID]),
                dcc.Store(id='tab2-card-title-data', data=f'{DEFAULT_DATA_DQN_NAME} vs {DEFAULT_DATA_FF_NAME}'),
            ]
        )
    ]
)
