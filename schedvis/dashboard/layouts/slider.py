import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html


def build_slider(tab, multiple=False):
    children = [
        dbc.Card([
            dbc.Row(id='app-controller', children=[
                dbc.Col(children=[
                    dbc.Row([
                        dbc.Col(dbc.Button(id=tab + 'pause-button', children='start', block=True)),
                        dbc.Col(dbc.Button(id=tab + 'reset-button', children='reset', block=True))
                    ])
                ], width=2),
                dbc.Col(
                    dcc.Upload(
                        id=tab + 'upload-data',
                        children=html.Div([
                            'Drag and Drop or ',
                            html.A('Select Files')
                        ]),
                        multiple=multiple,
                        style=dict(
                            width='100%',
                            textAlign='center'
                        )
                    ), width=2
                ),
                dbc.Col(
                    dcc.Slider(id=tab + 'slider', min=0, max=0, step=1, value=0, disabled=False),
                    width=7
                ),
                dbc.Col(html.H5(id=tab + 'slider-output-container', children='0'), width=1),


            ], align='center', justify='start', style=dict(padding='20px')),
            html.Br()
        ]),
        html.Br()
    ]
    return children
