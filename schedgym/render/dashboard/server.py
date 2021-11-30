from dash import Dash
from dash_bootstrap_components import themes
from dash_bootstrap_templates import load_figure_template

# setup bootstrap theme
load_figure_template('bootstrap')
external_stylesheets = [themes.JOURNAL]

app = Dash('server-dashboard', external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
server = app.server
