import threading
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import json
from flask import request
import requests
import random
import time
import plotly

###
# ubuntu run
# apt-get update
# apt-get install python3-pip
# python3 -m pip install dash
# git clone https://github.com/vkrot/tensortrade/
# cd tensortrade/tensortrade/vkrot
# python3 dash-server.py
# nohup python3 dash-server.py &

class FigureHolder():
    def __init__(self):
        self._figure = None

    def update_figure(self, figure):
        self._figure = figure

    def get_figure(self):
        return self._figure


class DashboardServer():
    def __init__(self, app: dash.Dash, figure_holder: FigureHolder):
        self.app = app
        self.figure_holder = figure_holder
        app.layout = html.Div(
            html.Div([
                html.H4('Simulation run:'),
                dcc.Graph(id='live-update-graph'),
                dcc.Interval(
                    id='interval-component',
                    interval=5000, # in milliseconds
                    n_intervals=0
                )
            ])
        )

        def endpoint():
            # figure_holder.update_figure(request.json)
            figure_holder.update_figure(plotly.io.from_json(request.json))
            # print(json.loads(request.json))
            return 'ok'

        app._add_url('update-fig', endpoint, methods=("POST",))

        @app.callback(Output('live-update-graph', 'figure'),
                      [Input('interval-component', 'n_intervals')])
        def update_graph_live(n):
            return figure_holder.get_figure()

    def run(self):
        self.app.run_server(debug=False, host="0.0.0.0")


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
figure_holder = FigureHolder()

server = DashboardServer(app, figure_holder)
threading.Thread(target=server.run, args=()).start()

# figure_holder.update_figure(fig)

# figure_holder.update_figure(plotly.io.from_json(s))


#
# while True:
#     import random
#     x = random.randint(1000, 10000)
#     requests.post('http://localhost:8050/update-fig', json=s)
#     time.sleep(1)
#


#     import plotly.graph_objects as go
#     fig = go.Figure(
#         data=[go.Bar(x=[1, 2, 3], y=[1, 3, random.randint(100, 200)])],
#         layout=go.Layout(
#             title=go.layout.Title(text=f"{time.time()} A Figure Specified By A Graph Object")
#         )
#     )
#
#     requests.post('http://localhost:8050/update-fig', json=fig.to_json())
#     time.sleep(1)