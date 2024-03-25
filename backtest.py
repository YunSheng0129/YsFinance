from .calendar import CALENDAR_TOOL
import numpy as np
import pandas as pd
from pandas import Series,DataFrame
from joblib import Parallel,delayed
from typing import Iterable
from math import log


def weights_limit(returns,weights,limit_up=True,limit_down=True):
    weights = weights.copy()
    if limit_up:
        i=0
        limit_loc = ((((returns.fillna(0.1) > 0.099)*weights).dropna(how='all').fillna(0))!=0)
        while True: 
            if i > 1000:
                raise ValueError('exceed the max iterations 1000, there are too many limit-up, please check your codes!')
            i+=1
            a = (limit_loc * weights.shift(1).fillna(0))
            b = (limit_loc * weights)
            c = b-a
            d = np.where(c>0,-c,0)
            if (d < 0).sum().sum() == 0:
                break
            weights = d + weights
    if limit_down:
        i=0
        limit_loc = ((((returns < -0.099)*weights).dropna(how='all').fillna(0))!=0)
        while True: 
            if i > 1000:
                raise ValueError('exceed the max iterations 1000, there are too many limit-down, please check your codes!')
            i+=1
            a = (limit_loc * weights.shift(1).fillna(0))
            b = (limit_loc * weights)
            c = b-a
            d = np.where(c<0,-c,0)
            if (d > 0).sum().sum() == 0:
                break
            weights = d + weights
    return weights




class QuickBackTestor:
    def __init__(self, start_date, end_date, prices):
        self.start_date = start_date
        self.end_date = end_date
        self._prices = prices
        self._returns = self._prices.pct_change().fillna(0)
        self._commission = None
        self._trade_dates = CALENDAR_TOOL.trade_date_in_range(self.start_date, self.end_date)
        self._results = {"strategy_rewards": None,"weights": None}

    def _rewards_in_range(self, start_position, start_date, end_date):
        returns_slice = self._returns.loc[start_date:end_date].copy(deep=True)
        # construct the imaginary returns:
        # short a stock is equal to long the stock that is purely negatively correlated with the prime stock
        returns_slice.iloc[:, np.where(start_position < 0)[0]] *= -1 
        start_position = np.abs(start_position)
        returns_slice.iloc[0] = 0
        pnls = (returns_slice + 1).cumprod()
        pnl = pnls.dot(start_position)
        rewards = pnl.pct_change()
        # pnl = np.dot(pnls,start_position)
        # pnl = pd.Series(pnl,index=pnls.index)
        rewards = pnl.pct_change()
        return rewards.iloc[1:]
    
    def ave_backtest(self):
        return (1+self._returns.mean(axis=1)).cumprod()
 
    def run_backtest(self, strategy: pd.DataFrame, limit_up=False,limit_down=False):
        strategy = weights_limit(self._returns,strategy,limit_up=limit_up,limit_down=limit_down)
        if not set(strategy.loc[self.start_date:self.end_date].index).issubset(set(self._trade_dates)):
            raise ValueError("the index of strategy must be a trade date")
        
        if strategy.isna().sum().sum() > 0:
            raise ValueError("the elements of startegy should not be nan")
        
        strategy = strategy.loc[self.start_date:self.end_date]
        rebalance_dates = strategy.index.to_list()
        dates = list(zip(rebalance_dates[:-1], rebalance_dates[1:]))
        results = []
        for start_date, end_date in dates:
            result = self._rewards_in_range(strategy.loc[start_date], start_date, end_date)
            results.append(result)
        self._results["strategy_rewards"] = pd.concat(results).reindex(self._trade_dates).fillna(0)
        self._results["weights"] = strategy
        return self._results["strategy_rewards"]
    
    def run_backtests(self, strategies: Iterable[pd.DataFrame]):
        """run multiple backtests
        The strategies should be aligned in the same format

        Parameters
        ----------
        strategies : Iterable[pd.DataFrame]
            _description_
        """
        _strategies = []
        for strategy in strategies:
            _strategies.append(strategy.loc[self.start_date:self.end_date])
        sample_strategy = _strategies[0]
        if not set(sample_strategy.index).issubset(set(self._trade_dates)):
            raise ValueError("the index of strategy must be a trade date")
        
        if sample_strategy.isna().sum().sum() > 0:
            raise ValueError("the elements of startegy should not be nan")
        
        rebalance_dates = sample_strategy.index.to_list()
        dates = list(zip(rebalance_dates[:-1], rebalance_dates[1:]))
        results = [] 
        for start_date, end_date in dates:
            current_strategy = pd.concat([strategy.loc[start_date] for strategy in strategies], axis=1, ignore_index=True)
            result = self._rewards_in_range(current_strategy, start_date, end_date)
            results.append(result)
        self._results["strategy_rewards"] = pd.concat(results).reindex(self._trade_dates).fillna(0)  
        return self._results["strategy_rewards"]

    def reset(self):
        self._results = {"strategy_rewards": None}

    def summary(self):
        pass

    

class DifferBackTestor:
    '''
    For this class you should give the MultiIndex weights Series with date and assets
    '''
    
    def __init__(self, start_date, end_date, prices):
        self.start_date = start_date
        self.end_date = end_date
        self._prices = prices
        self._assets = prices.columns
        self._returns = self._prices.pct_change().fillna(0)
        self._trade_dates = CALENDAR_TOOL.trade_date_in_range(self.start_date, self.end_date)

    def _rewards_in_range(self, start_position, start_date, end_date):
        returns_slice = self._returns.loc[start_date:end_date].copy(deep=True)
        # construct the imaginary returns:
        # short a stock is equal to long the stock that is purely negatively correlated with the prime stock
        returns_slice.iloc[:, np.where(start_position < 0)[0]] *= -1 
        start_position = np.abs(start_position)
        returns_slice.iloc[0] = 0
        pnls = (returns_slice + 1).cumprod()
        pnl = pnls.dot(start_position)
        rewards = pnl.pct_change()
        rewards = pnl.pct_change()
        return rewards.iloc[1:]
    def run_backtest(self,weights:Series):
        weights = weights.loc[self.start_date:self.end_date].copy()
        dates = weights.index.get_level_values(0).unique().sort_values()
        dates_range = zip(dates[0:-1],dates[1:])
        def rewards(date):
            _start_position = weights.loc[date[0]].reindex(self._assets).fillna(0)
            return self._rewards_in_range(_start_position,date[0],date[1])
        works_to_do = [delayed(rewards)(date) for date in dates_range]
        results = Parallel(n_jobs=8)(works_to_do)
        return pd.concat(results)
    

