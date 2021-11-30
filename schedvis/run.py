import logging

from dashboard.app import app

def main():
    app.logger.setLevel(logging.DEBUG)
    # https://github.com/plotly/dash/issues/532
    app.logger.setLevel = lambda x: None
    app.run_server(debug=True)
    
