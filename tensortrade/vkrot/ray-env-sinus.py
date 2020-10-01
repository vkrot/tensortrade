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
    x = np.arange(0, 2 * np.pi, 2 * np.pi / 1001)
    y = 50 * np.sin(3 * x) + 100

    x = np.arange(0, 2 * np.pi, 2 * np.pi / 1000)
    p = Stream.source(y, dtype="float").rename("USD-TTC")

    coinbase = Exchange("coinbase", service=execute_order)(
        p
    )

    cash = Wallet(coinbase, 100000 * USD)
    asset = Wallet(coinbase, 0 * TTC)

    portfolio = Portfolio(USD, [
        cash,
        asset
    ])

    feed = DataFeed([
        p,
        p.rolling(window=10).mean().rename("fast"),
        p.rolling(window=50).mean().rename("medium"),
        p.rolling(window=100).mean().rename("slow"),
        p.log().diff().fillna(0).rename("lr")
    ])

    reward_scheme = PBR(price=p)

    action_scheme = BSH(
        cash=cash,
        asset=asset
    ).attach(reward_scheme)

    renderer_feed = DataFeed([
        Stream.source(y, dtype="float").rename("price"),
        Stream.sensor(action_scheme, lambda s: s.action, dtype="float").rename("action")
    ])

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

env = create_env({
"window_size": 25
})
done = False
obs = env.reset()
while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
