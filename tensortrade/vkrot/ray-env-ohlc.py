### COMMAND

from tensortrade.oms.instruments import Instrument

USD = Instrument("USD", 2, "U.S. Dollar")
TTC = Instrument("TTC", 8, "TensorTrade Coin")

### COMMAND

from gym.spaces import Discrete

from tensortrade.env.default.actions import TensorTradeActionScheme

from tensortrade.env.generic import ActionScheme, TradingEnv
from tensortrade.core import Clock
from tensortrade.oms.instruments import ExchangePair
from tensortrade.oms.wallets import Portfolio
from tensortrade.env.default.renderers import PlotlyTradingChart
from tensortrade.oms.orders import (
    Order,
    proportion_order,
    TradeSide,
    TradeType
)


class BSH(TensorTradeActionScheme):

    registered_name = "bsh"

    def __init__(self, cash: 'Wallet', asset: 'Wallet'):
        super().__init__()
        self.cash = cash
        self.asset = asset

        self.listeners = []
        self.action = 0

    @property
    def action_space(self):
        return Discrete(2)

    def attach(self, listener):
        self.listeners += [listener]
        return self

    def get_orders(self, action: int, portfolio: 'Portfolio'):
        order = None

        if abs(action - self.action) > 0:
            src = self.cash if self.action == 0 else self.asset
            tgt = self.asset if self.action == 0 else self.cash
            order = proportion_order(portfolio, src, tgt, 1.0)
            self.action = action

        for listener in self.listeners:
            listener.on_action(action)

        return [order]

    def reset(self):
        super().reset()
        self.action = 0


### COMMAND

from tensortrade.env.default.rewards import TensorTradeRewardScheme
from tensortrade.feed.core import Stream, DataFeed


class PBR(TensorTradeRewardScheme):

    registered_name = "pbr"

    def __init__(self, price: 'Stream'):
        super().__init__()
        self.position = -1

        r = Stream.sensor(price, lambda p: p.value, dtype="float").diff()
        position = Stream.sensor(self, lambda rs: rs.position, dtype="float")

        reward = (r * position).fillna(0).rename("reward")

        self.feed = DataFeed([reward])
        self.feed.compile()

    def on_action(self, action: int):
        self.position = -1 if action == 0 else 1

    def get_reward(self, portfolio: 'Portfolio'):
        return self.feed.next()["reward"]

    def reset(self):
        self.position = -1
        self.feed.reset()

### COMMAND

import matplotlib.pyplot as plt

from tensortrade.env.generic import Renderer


class PositionChangeChart(Renderer):

    def __init__(self, color: str = "orange"):
        self.color = "orange"

    def render(self, env, **kwargs):
        history = pd.DataFrame(env.observer.renderer_history)

        actions = list(history.action)
        p = list(history.price)

        buy = {}
        sell = {}

        for i in range(len(actions) - 1):
            a1 = actions[i]
            a2 = actions[i + 1]

            if a1 != a2:
                if a1 == 0 and a2 == 1:
                    buy[i] = p[i]
                else:
                    sell[i] = p[i]

        buy = pd.Series(buy)
        sell = pd.Series(sell)

        fig, axs = plt.subplots(1, 2, figsize=(15, 5))

        fig.suptitle("Performance")

        axs[0].plot(np.arange(len(p)), p, label="price", color=self.color)
        axs[0].scatter(buy.index, buy.values, marker="^", color="green")
        axs[0].scatter(sell.index, sell.values, marker="^", color="red")
        axs[0].set_title("Trading Chart")

        env.action_scheme.portfolio.performance.plot(ax=axs[1])
        axs[1].set_title("Net Worth")

        plt.show()


### COMMAND

import ray
import numpy as np
import pandas as pd

from ray import tune
from ray.tune.registry import register_env

import tensortrade.env.default as default

from tensortrade.feed.core import DataFeed, Stream
from tensortrade.oms.exchanges import Exchange
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.oms.wallets import Wallet, Portfolio
from tensortrade.env.default.renderers import PlotlyTradingChart, FileLogger
from tensortrade.env.default.callbacks import LoggingCallback


def create_env(config):
    cdd = CryptoDataDownload()
    data = cdd.fetch("Coinbase", "USD", "BTC", "1h")

    features = []
    for c in data.columns[1:]:
        s = Stream.source(list(data[c]), dtype="float").rename(data[c].name)
        features += [s]

    cp = Stream.select(features, lambda s: s.name == "close")

    features = [
        cp
    ]
    feed = DataFeed(features)
    feed.compile()

    coinbase = Exchange("coinbase", service=execute_order)(
        Stream.source(list(data["close"]), dtype="float").rename("USD-BTC")
    )

    cash = Wallet(coinbase, 100000 * USD)
    asset = Wallet(coinbase, 0 * TTC)

    portfolio = Portfolio(USD, [
        cash,
        asset
    ])

    renderer_feed = DataFeed([
        Stream.source(list(data["date"])).rename("date"),
        Stream.source(list(data["open"]), dtype="float").rename("open"),
        Stream.source(list(data["high"]), dtype="float").rename("high"),
        Stream.source(list(data["low"]), dtype="float").rename("low"),
        Stream.source(list(data["close"]), dtype="float").rename("close"),
        Stream.source(list(data["volume"]), dtype="float").rename("volume")
    ])

    reward_scheme = rewards.SimpleProfit()
    action_scheme = actions.BSH(cash, asset)

    chart_renderer = PlotlyTradingChart(
        display=True,  # show the chart on screen (default)
        height=800,  # affects both displayed and saved file height. None for 100% height.
        save_format="html",  # save the chart to an HTML file
        auto_open_html=True,  # open the saved HTML chart in a new browser tab
    )

    import uuid
    uid = uuid.uuid4()

    callback = LoggingCallback(f'/Users/vkrot/callback-{uid}', chart_renderer)

    environment = default.create(
        feed=feed,
        portfolio=portfolio,
        action_scheme=action_scheme,
        reward_scheme=reward_scheme,
        renderer_feed=renderer_feed,
        # renderer=PositionChangeChart(),
        renderer=PositionChangeChart(),
        window_size=config["window_size"],
        max_allowed_loss=0.6,
        callback=callback
    )
    return environment


register_env("TradingEnv", create_env)


### COMMAND

# analysis = tune.run(
#     "PPO",
#     stop={
#       "episode_reward_mean": 500
#     },
#     config={
#         "env": "TradingEnv",
#         "env_config": {
#             "window_size": 25
#         },
#         "log_level": "DEBUG",
#         "framework": "tf2",
#         "ignore_worker_failures": True,
#         "num_workers": 3,
#         "num_gpus": 0,
#         "clip_rewards": True,
#         "lr": 8e-6,
#         "lr_schedule": [
#             [0, 1e-1],
#             [int(1e2), 1e-2],
#             [int(1e3), 1e-3],
#             [int(1e4), 1e-4],
#             [int(1e5), 1e-5],
#             [int(1e6), 1e-6],
#             [int(1e7), 1e-7]
#         ],
#         "gamma": 0,
#         "observation_filter": "MeanStdFilter",
#         "lambda": 0.72,
#         "vf_loss_coeff": 0.5,
#         "entropy_coeff": 0.01
#     },
#     checkpoint_at_end=True
# )

import tensortrade.env.default as default
from tensortrade.data.cdd import CryptoDataDownload
from tensortrade.feed.core import Stream, DataFeed
from tensortrade.oms.exchanges import Exchange
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.oms.instruments import USD, BTC, ETH
from tensortrade.oms.wallets import Wallet, Portfolio
from tensortrade.agents import DQNAgent
import tensortrade.env.default.actions as actions
import tensortrade.env.default.rewards as rewards
import tensortrade.env.default.stoppers as stoppers
import threading


def build_env():

    cdd = CryptoDataDownload()
    data = cdd.fetch("Coinbase", "USD", "BTC", "1h")

    features = []
    for c in data.columns[1:]:
        s = Stream.source(list(data[c]), dtype="float").rename(data[c].name)
        features += [s]

    cp = Stream.select(features, lambda s: s.name == "close")

    features =[
        cp
    ]
    feed = DataFeed(features)
    feed.compile()

    coinbase = Exchange("coinbase", service=execute_order)(
        Stream.source(list(data["close"]), dtype="float").rename("USD-BTC")
    )

    cash = Wallet(coinbase, 10000 * USD)
    asset = Wallet(coinbase, 0 * BTC)

    portfolio = Portfolio(USD, [
        cash,
        asset
    ])

    renderer_feed = DataFeed([
        Stream.source(list(data["date"])).rename("date"),
        Stream.source(list(data["open"]), dtype="float").rename("open"),
        Stream.source(list(data["high"]), dtype="float").rename("high"),
        Stream.source(list(data["low"]), dtype="float").rename("low"),
        Stream.source(list(data["close"]), dtype="float").rename("close"),
        Stream.source(list(data["volume"]), dtype="float").rename("volume")
    ])

    reward_scheme = rewards.SimpleProfit()
    action_scheme = actions.BSH(cash, asset)


    plotly = PlotlyTradingChart(
        display=True,
        height=700,
        save_format="html"
    )
    callback = LoggingCallback('http://165.227.193.153:8050', plotly)

    env = default.create(
        portfolio=portfolio,
        action_scheme=action_scheme,
        reward_scheme=reward_scheme,
        feed=feed,
        renderer_feed=renderer_feed,
        renderer=plotly,
        window_size=20,
        callback=callback
    )
    return env


# env = create_env({
#     "window_size": 25
# })

env = build_env()

done = False
obs = env.reset()
while True:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    # if done:
    #     print('step done')
    #     env.reset()
    # else:
    #     print('no')
