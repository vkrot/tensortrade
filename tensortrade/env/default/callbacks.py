from abc import abstractmethod
import requests
from tensortrade.env.default.renderers import PlotlyTradingChart
import threading
from tensortrade.core.component import Component
from tensortrade.core.base import TimeIndexed
from tensortrade.env.generic import EpisodeCallback
import time

class LoggingCallback(EpisodeCallback):
    def __init__(self, host: str, plotly_renderer: PlotlyTradingChart):
        self.host = host
        self.plotly_renderer = plotly_renderer
        self._fig = None
        thr = threading.Thread(target=self.update, args=(), daemon=True)
        thr.start()

    def update(self):
        while True:
            try:
                requests.post(f'{self.host}/update-fig', json=self._fig)
            except Exception as ex:
                print(f'Error: {ex}')
            time.sleep(5)

    def on_done(self, env: 'TradingEnv') -> None:
        self.plotly_renderer.render(env)
        self._fig = self.plotly_renderer.fig.to_json()