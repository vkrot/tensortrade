### COMMAND

from gym.spaces import Discrete
import numpy as np
from tensortrade.env.default.actions import TensorTradeActionScheme

from tensortrade.env.generic import ActionScheme, TradingEnv
from tensortrade.core import Clock
from tensortrade.oms.instruments import ExchangePair
from tensortrade.oms.wallets import Portfolio
from tensortrade.env.default.callbacks import LoggingCallback
from tensortrade.env.default.renderers import PlotlyTradingChart
from tensortrade.oms.orders import (
    Order,
    proportion_order,
    TradeSide,
    TradeType
)

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
from ray import tune
from ray.tune.registry import register_env
from ray.tune.logger import DEFAULT_LOGGERS
from ray.tune.integration.wandb import WandbLogger
import ray
from ray.rllib.agents.ppo import PPOTrainer
import tensorflow as tf
import talib
from tensortrade.env.default.rewards import TensorTradeRewardScheme, SimpleProfit

tf.keras.backend.set_floatx('float32')

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


class PBR(TensorTradeRewardScheme):

    registered_name = "pbr"

    def __init__(self, rsi: Stream, rsi_diff: Stream):
        super().__init__()
        self.position = -1

        # r = Stream.sensor(rsi, lambda p: p.value, dtype="float").diff()
        # position = Stream.sensor(self, lambda rs: rs.position, dtype="float")
        #
        # reward = (r * position).fillna(0).rename("reward")
        #
        # self.feed = DataFeed([reward])
        # self.feed.compile()
        self.rsi = rsi
        self.rsi_diff = rsi_diff

    def on_action(self, action: int):
        self.position = -1 if action == 0 else 1

    def get_reward(self, portfolio: 'Portfolio'):
        # return self.feed.next()["reward"]
        # print(f'rsi: {self.rsi.value} positition: {self.position}')

        r = self.rsi.value
        rd = self.rsi_diff.value
        if np.isnan(r):
            return 0

        if rd > 0:
            # buy low rsi values
            return self.position * -1 * (1 - r)
        else:
            # sell high rsi values
            return self.position * r

        # if r < 0.3:
        #     return self.position * -1
        # else:
        #     return self.position

    def reset(self):
        self.position = -1
        # self.feed.reset()


def build_env(config):
    worker_index = 1
    if hasattr(config, 'worker_index'):
        worker_index = config.worker_index

    cdd = CryptoDataDownload()
    data = cdd.fetch("Coinbase", "USD", "BTC", "1h")

    features = []
    for c in data.columns[1:]:
        s = Stream.source(list(data[c]), dtype="float").rename(data[c].name)
        features += [s]

    # cp = Stream.select(features, lambda s: s.name == "close")
    # features =[
    #     cp
    # ]

    rsi_indicator = talib.RSI(data['close'], 14)
    rsi_indicator = rsi_indicator / 100
    rsi: Stream = Stream.source(rsi_indicator, "float").rename("rsi")
    rsi_diff = rsi.diff().rename("rsi_diff")

    features = [rsi, rsi_diff]

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

    # reward_scheme = rewards.SimpleProfit()
    reward_scheme = PBR(rsi=rsi, rsi_diff=rsi_diff)
    # reward_scheme = SimpleProfit(window_size=25)
    action_scheme = BSH(cash, asset)
    action_scheme.attach(reward_scheme)

    plotly = PlotlyTradingChart(
        display=True,
        height=700,
        save_format="html"
    )
    env = default.create(
        portfolio=portfolio,
        action_scheme=action_scheme,
        reward_scheme=reward_scheme,
        feed=feed,
        renderer_feed=renderer_feed,
        renderer=plotly,
        window_size=20,
        max_allowed_loss=0.5,
        callback=(LoggingCallback('http://165.227.193.153:8050', plotly) if worker_index == 1 else None)
    )

    import logging
    import os
    LOGGER = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s [%(threadName)s] - %(levelname)s - %(message)s',
    )
    LOGGER.info('env created logger')
    LOGGER.info(f'env: {os.environ}')
    print(f'env: {os.environ}')
    print('env created')

    return env


register_env("TradingEnv", build_env)

trainer_config = {
    "env": "TradingEnv",
    "env_config": {
        "window_size": 25
    },
    "log_level": "DEBUG",
    "framework": "tf2",
    "ignore_worker_failures": True,
    "num_workers": 3,
    "num_gpus": 0,
    "clip_rewards": True,
    "lr": 8e-6,
    "lr_schedule": [
        [0, 1e-1],
        [int(1e2), 1e-2],
        [int(1e3), 1e-3],
        [int(1e4), 1e-4],
        [int(1e5), 1e-5],
        [int(1e6), 1e-6],
        [int(1e7), 1e-7]
    ],
    "gamma": 0.9,
    "observation_filter": "MeanStdFilter",
    "lambda": 0.72,
    "vf_loss_coeff": 0.5,
    "entropy_coeff": 0.01,
    "logger_config": {
        "wandb": {
            "project": "ray_0",
            "api_key": "14343327bef59b042b78cf109a468f990c0d3d95",
            "log_config": True
        }
    }
}

analysis = tune.run(
    "PPO",
    stop={
      "episode_reward_mean": 500
    },
    config=trainer_config,
    loggers=DEFAULT_LOGGERS + (WandbLogger, ),
    checkpoint_at_end=True
)

# debug code
# ray.init(num_gpus=1, local_mode=True)
# agent = PPOTrainer(
#     env="TradingEnv",
#     config=trainer_config
# )
# agent.train()


# compute final reward
# ray.init(num_gpus=1, local_mode=False)
# env = build_env({
#     "window_size": 25
# })
# episode_reward = 0
# done = False
# obs = env.reset()
#
# while not done:
#     action = agent.compute_action(obs)
#     obs, reward, done, info = env.step(action)
#     episode_reward += reward
# print(f'reward: {episode_reward}')