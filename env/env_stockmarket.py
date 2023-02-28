from __future__ import annotations

import gym
import numpy as np
import pandas as pd

import config

from env.state import State

from gym import spaces
from gym.utils import seeding

class VirtualStockmarket(gym.Env):

    def __init__(self,
        data: pd.DataFrame,
        data_close_column: str,

        initial_account_balance: float,
        buy_cost: float,
        sell_cost: float,

        hmax: float,
        reward_scaling: float,
        turbulence_threshold: float,

        state_exclude_features: list[str],
        state_previous: State | None = None,
        data_extra: pd.DataFrame = pd.DataFrame()
    ):

        self.data                       = data.sort_values(by=[data.index.name, str(config.DataColumnNames.TICKER)])
        self.data_close_column          = data_close_column
        self.data_extra                 = data_extra
        self.data_sample_index          = 0

        self.buy_cost                   = buy_cost
        self.sell_cost                  = sell_cost

        self.hmax                       = hmax
        self.reward_scaling             = reward_scaling
        self.turbulence_threshold       = turbulence_threshold

        self.num_stocks                 = len(self.data[str(config.DataColumnNames.TICKER)].unique())

        self.state_exclude_features     = state_exclude_features

        self.state                      = State(
        # state space description (data layout: labeled index + dimension)
        {
            str(config.DataColumnNames.BALANCE):    1,
            'num_shares':                           self.num_stocks,

            **dict([(feature, self.num_stocks)  for feature in self.data.columns if feature not in self.state_exclude_features]),
            **dict([(feature, 1)                for feature in self.data_extra.columns]),
        },
        # default/initial state values
        {
            str(config.DataColumnNames.BALANCE):    initial_account_balance,
            'num_shares':                           np.zeros(self.num_stocks),

            **dict([(feature, self.data.loc[self.data.index.unique()[0]][feature].values if self.num_stocks > 1 else [self.data.iloc[0][feature]]) for feature in self.data.columns if feature not in self.state_exclude_features]),
            **dict([(feature, self.data_extra.loc[self.data.index.unique()[0]][feature]) for feature in self.data_extra.columns]),
        })

        # define openai.gym action and state space
        if config.ENV_NORMALIZE_ACTIONS:
            self.action_space           = spaces.Box(low=-1,         high=1,         shape=(self.num_stocks,))
        else: 
            self.action_space           = spaces.Box(low=-self.hmax, high=self.hmax, shape=(self.num_stocks,))
 
        self.observation_space          = spaces.Box(low=-np.inf,    high=np.inf,    shape=(len(self.state),))
        
        self.last_total_assets          = 0
        self.assets_book                = [[] for _ in range(self.num_stocks)]

        self.reset()

    def _total_asset_value(self):
        return float(self.state[str(config.DataColumnNames.BALANCE)]) + np.dot(self.state.num_shares, self.state[self.data_close_column])

    def _sell_stock(self, stock_index, amount):
        # only attempt to sell shares, if we currently have any shares in it and it is traded at this particular time
        if self.state.num_shares[stock_index] > 0 and self.state[self.data_close_column][stock_index] > 0:
            sell_shares                                         = min(amount, self.state.num_shares[stock_index])

            # place order ...
            # we will assume al order will be processed instantly.
            value                                               = self.state[self.data_close_column][stock_index] * sell_shares
            costs                                               = value * self.sell_cost
            sold_value                                          = value - costs
            sold_shares                                         = sell_shares

            # update account balance and shares
            self.state[str(config.DataColumnNames.BALANCE)]     += sold_value
            self.state.num_shares[stock_index]                  -= sold_shares

            return { 'shares': sold_shares, 'price': self.state[self.data_close_column][stock_index] * (1.0 - self.buy_cost), 'action': 'SELL' }

        return None

    def _buy_stock(self, stock_index, amount):
        # only attempt to buy shares, if it is traded at this particular time
        if self.state[self.data_close_column][stock_index] > 0:
            # prevent buying to many shares, if account balance is insufficient
            available_funds                                     = self.state[str(config.DataColumnNames.BALANCE)] // (self.state[self.data_close_column][stock_index] * (1.0 + self.buy_cost))  
            buy_shares                                          = int(min(available_funds, amount))

            if buy_shares:
                # place order ...
                # we will assume al order will be processed instantly.
                value                                               = self.state[self.data_close_column][stock_index] * buy_shares
                costs                                               = value * self.buy_cost
                bought_value                                        = value + costs
                bought_shares                                       = buy_shares

                # update account balance and shares
                self.state[str(config.DataColumnNames.BALANCE)]     -= bought_value
                self.state.num_shares[stock_index]                  += bought_shares

                return { 'shares': bought_shares, 'price': self.state[self.data_close_column][stock_index] * (1.0 + self.buy_cost), 'action': 'BUY' }

        return None

    def _do_trade(self, actions):
        trades      = [None] * self.num_stocks

        if config.ENV_NORMALIZE_ACTIONS:
            # actions initially is scaled between -1 to 1
            actions = actions * self.hmax
        # convert into integer because we can't by fraction of shares
        actions = actions.astype(int) 

        # if current market turbulence is above threshold, force sell all shares!
        if self.state.has(str(config.DataColumnNames.TURBULENCE)) and float(self.state[str(config.DataColumnNames.TURBULENCE)]) >= self.turbulence_threshold:
            actions = np.array(-self.state['num_shares'])

        # sell actions
        for stock_index in np.where(actions < 0)[0]:
            sold = self._sell_stock(stock_index, abs(actions[stock_index]))
            if sold:
                trades[stock_index] = sold

        # buy actions
        for stock_index in np.where(actions > 0)[0]:
            bought = self._buy_stock(stock_index, abs(actions[stock_index]))
            if bought:
                trades[stock_index] = bought

        return trades

    def _update_asset_book(self, trades: list(dict)) -> list[float]:
        profits_losses                          = np.zeros(len(trades))
        for stock_index, trade                  in enumerate(trades):
            profits_losses[stock_index]         = 0.0
            if trade:
                if trade['action']              == 'BUY':
                    self.assets_book[stock_index].append(trade)
                elif trade['action']            == 'SELL':
                    quantity                    = trade['shares']
                    sold_value                  = quantity * trade['price']

                    purc_value                  = 0.0
                    while quantity              > 0:
                        last_asset_book_entry   = self.assets_book[stock_index][-1]
                        if quantity             < last_asset_book_entry['shares']:
                            purc_value          = purc_value + (quantity * last_asset_book_entry['price'])
                            self.assets_book[stock_index][-1]['shares'] = last_asset_book_entry['shares'] - quantity
                            quantity            = 0
                        else: # greater or equal
                            self.assets_book[stock_index].pop()
                            purc_value          = purc_value + (last_asset_book_entry['shares'] * last_asset_book_entry['price'])
                            quantity            = quantity - last_asset_book_entry['shares']

                    profits_losses[stock_index] = sold_value - purc_value
                else:
                    raise Exception(f"Invalid trade action '{trade['action']}'.")

        return profits_losses

    def step(self, actions):
        reward  = 0
        done    = self.data_sample_index >= (len(self.data.index.unique()) - 1)
        state   = self.state.copy()
        extra   = { 
            'time_step' :           self.data.index.unique()[self.data_sample_index],
            'last_state':           state,
            'trades'    :           [None] * self.num_stocks,
            'profit_loss':          0.0
        }

        if done:
            pass
        else:
            extra['trades']         = self._do_trade(actions)
            profits_losses          = self._update_asset_book(extra['trades'])
            total_asset_value       = self._total_asset_value()

            # compute reward
            reward_profits          = \
            extra['profit_loss']    = sum(profits_losses)
            reward_total_assets     = total_asset_value - self.last_total_assets
            reward                  = np.dot(
                [
                    reward_total_assets, 
                    reward_profits
                ],
                # weights
                [
                    0.8,
                    0.2
                ]
            )

            # if we run out of funds the game is done
            # if not np.any(np.less_equal(self.state[self.data_close_column], self.state[str(config.DataColumnNames.BALANCE)][0])):
            #     done = True

            # update state
            self.data_sample_index  += 1
            self.state.update({
                **dict([(feature, self.data.loc[self.data.index.unique()[self.data_sample_index]][feature].values if self.num_stocks > 1 else [self.data.iloc[self.data_sample_index][feature]]) for feature in self.data.columns if feature not in self.state_exclude_features]),
                **dict([(feature, self.data_extra.loc[self.data.index.unique()[self.data_sample_index]][feature]) for feature in self.data_extra.columns]),
            })
            self.last_total_assets  = total_asset_value

        return state._data, reward * (config.ENV_REWARD_SCALING if config.ENV_REWARD_SCALING else 1.0), done, extra

    def reset(self, env_reset_states: dict = {}):
        self.data_sample_index      = 0

        self.state.reset()
        self.last_total_assets      = self._total_asset_value()
        self.assets_book            = [[] for _ in range(self.num_stocks)]

        for field, value in env_reset_states.items():
            if value is not None:
                setattr(self, field, value)

        return self.state._data

    def render(self, mode="human", close=False):
        return self.state

    def close(self):
        pass

    def seed(self, seed=None):
        _, seed = seeding.np_random(seed)
        return [seed]