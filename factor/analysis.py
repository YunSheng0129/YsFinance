from .utils import get_clean_factor_and_forward_returns
from ..tools import DataHandle
from ..backtest import QuickBackTestor
from pandas import Series,DataFrame
import pandas as pd
import numpy as np
from ..data_statistics import StatTtest
from .utils import MaxLossExceededError



def factor_neutral_groupby(factor:DataFrame,group:dict,names=['date','group']):
    '''按组和日期对因子数据进行标准化，需要的时间较长，记得跑完存盘'''
    factor = factor.copy(deep=True)
    factor['group'] = factor.index.get_level_values(level=1).map(group)
    return factor.groupby(names).transform(DataHandle.normalize)

class SingleFactorAnalysis:

    def __init__(self,start_date,end_date,factor:Series,prices:DataFrame,periods=(1,5,20),quantiles=5,groupby=None,periods_by_factor=True):
        self.start_date = start_date
        self.end_date = end_date
        self._prices = prices.loc[start_date:end_date]
        self._factor = factor.loc[start_date:end_date]
        self.universe = prices.columns
        self.n_assets = len(self.universe)
        self._group_by = groupby
        self._periods = periods
        self._quantiles = quantiles
        self.periods_by_factor = periods_by_factor
        self.backtest = QuickBackTestor(self.start_date,self.end_date,self._prices)
        self._cleaned_factor = get_clean_factor_and_forward_returns(factor=self._factor,
                                                                    prices=self._prices,
                                                                    periods=periods,
                                                                    quantiles=quantiles,
                                                                    groupby=groupby,
                                                                    filter_zscore=None,
                                                                    zero_aware=False,
                                                                    periods_by_factor=periods_by_factor)
        

    def IC(self, look_forward=None, method="spearman"):
        if self.periods_by_factor:
            look_forward = '1T'
        else:
            if look_forward is None:
                raise ValueError('look_forward should not be None if you do not set periods_by_factor.')
            look_forward = f"{look_forward}D"
        return self._cleaned_factor.groupby("date").apply(lambda x: x[look_forward].corr(x["factor"], method=method))
    
    def cumIC(self,look_forward=None, method="spearman"):
        return self.IC(look_forward,method).cumsum()    
    
    def get_group_assets(self,factor_quantile):
        tmp = self._cleaned_factor[self._cleaned_factor['factor_quantile'] == factor_quantile]["factor_quantile"]
        return tmp
    
    def get_group_positions(self, factor_quantile: int):
        tmp = self._cleaned_factor[self._cleaned_factor['factor_quantile'] == factor_quantile]["factor_quantile"].unstack().reindex(self.universe, axis=1)
        weights = (tmp > 0) / np.repeat((tmp>0).sum(axis=1).values.reshape(-1,1), self.n_assets, axis=1)
        return weights
    
    def get_long_short_positions(self):
        weights1 = self.get_group_positions(factor_quantile=1)
        weights2 = self.get_group_positions(factor_quantile=self._quantiles)
        return weights2/2-weights1/2
    
    def group_return(self,factor_quantile:int,limit_up=True,limit_down=True):
        weights = self.get_group_positions(factor_quantile)
        return self.backtest.run_backtest(weights,limit_up,limit_down)
    
    def groups_return(self,limit_up=True,limit_down=True):
        """这里是returns,不是PnL,懒得改"""
        PnLs = []
        for i in range(1,self._quantiles + 1):
            PnL = self.group_return(factor_quantile=i,limit_up=limit_up,limit_down=limit_down)
            PnLs.append(PnL)
        PnLs = pd.concat(PnLs,axis=1)
        PnLs.columns = [f'group{i}' for i in range(1,self._quantiles + 1)]
        return PnLs
    
    def long_short_return(self,limit_up=True,limit_down=True):
        weights = self.get_long_short_positions()
        return self.backtest.run_backtest(weights,limit_up,limit_down)
    
    def hypo_testing(self,look_forward=None,method='RLM',alpha=0.05):
        if self.periods_by_factor:
            look_forward = '1T'
        else:
            if look_forward is None:
                raise ValueError('look_forward should not be None if you do not set periods_by_factor.')
            look_forward = f"{look_forward}D"
        test_bool = []
        for date in self._cleaned_factor.index.get_level_values(0).drop_duplicates():
            X = self._cleaned_factor.loc[date]['factor']
            y = self._cleaned_factor.loc[date][look_forward]
            test = StatTtest(X,y,method=method)
            test_bool.append(test.t_test(alpha=alpha))
        test_frame = DataFrame(np.array(test_bool))
        if len(test_frame.columns) != 3:
            print(test_frame) 
            print(self._cleaned_factor)
        test_frame.columns = ['beta','u','d']
        total = len(test_frame.index)
        if total == 0:
            return Series(np.zeros(8),
                          index = ['正向显著比例','负向显著比例','正-负','同向显著次数比','状态切换次数比','同向-切换','因子方向','总检验次数'])
        u = test_frame['u'].sum()
        d = test_frame['d'].sum()
        pass_frame = test_frame[(test_frame['u']==True)|(test_frame['d']==True)]
        sep = pd.concat([pass_frame['beta'],pass_frame['u'].shift(1)],axis=1).dropna()
        sep.columns = ['0','1']
        same = (sep['0'] == sep['1']).sum()
        dif = (sep['0'] != sep['1']).sum()
        if u>=d:
            direct = 1
        else:
            direct = -1
        return Series([u/total,d/total,(u-d)/total, same/total,dif/total,(same-dif)/total,direct,total],
                      index = ['正向显著比例','负向显著比例','正-负','同向显著次数比','状态切换次数比','同向-切换','因子方向','总检验次数'])       


class MultiFactorAnalysis:
    """
    This class allow you to do multi-factor analysis, you can input factor DataFrame. But if your file is too big, it could be slow.

    Params:
    -start_date: str|datetime.
    -end_date: str|datetime.
    -factor:DataFrame. MultiIndex with (date,assets), columns should be factor name.
    -prices:DataFrame. index should be datetime, columns should be assets name(same as factor assets).
    -periods:Any=(1,5,20). the time periods you calculate the forward returns, it follows the factor frequency! The cleand factor frame -columns will be '1D','5D','20D'. 
    -quantiles:int=5. the group number you seperate the factor every date.
    -groupby:dict=None. keys are assets name, values are group name, usually industry map.
    -periods_by_factor:bool=True. True, then the periods will be related with factor frequency, and periods input will be useless.

    Attention:
        None
    """
    def __init__(self,start_date,end_date,factor:DataFrame,prices:DataFrame,periods=(1,5,20),quantiles=5,groupby=None,periods_by_factor=True):
        self.start_date = start_date
        self.end_date = end_date
        self._prices = prices.loc[start_date:end_date]
        self._factor = factor.loc[start_date:end_date]
        self.universe = prices.columns
        self.n_assets = len(self.universe)
        self._group_by = groupby
        self._periods = periods
        self._quantiles = quantiles
        self.periods_by_factor = periods_by_factor
        self._factor_name = self._factor.columns
        self.backtest = QuickBackTestor(self.start_date,self.end_date,self._prices)
        cleaned_factors = {}
        wrong_fac = []
        for fac in self._factor_name:
            try:
                cleaned_factors[fac] = get_clean_factor_and_forward_returns(factor=self._factor[fac],
                                                                        prices=self._prices,
                                                                        periods=periods,
                                                                        quantiles=quantiles,
                                                                        groupby=groupby,
                                                                        filter_zscore=None,
                                                                        zero_aware=False,
                                                                        periods_by_factor=periods_by_factor)
            except MaxLossExceededError:
                wrong_fac.append(fac)
                continue
        print('You have wrong factor which max_loss exceed 35%, these have been deleted, please check.')
        print(wrong_fac)
        self._factor_name = list(set(self._factor_name).difference(wrong_fac))
        self._wrong_factor = wrong_fac
        self._cleaned_factors = cleaned_factors
        del self._factor

    
    def multi_ICs(self, fac_name:list=None,look_forward=None, method="spearman"):
        if fac_name is None:
            fac_name = self._factor_name

        if self.periods_by_factor:
            look_forward = '1T'
        else:
            if look_forward is None:
                raise ValueError('look_forward should not be None if you do not set periods_by_factor.')
            look_forward = f"{look_forward}D"

        IC_lst = []
        for fac in fac_name:
            cf = self._cleaned_factors[fac]
            IC_lst.append(cf.groupby("date").apply(lambda x: x[look_forward].corr(x["factor"], method=method)))
        ICs = pd.concat(IC_lst,axis=1)
        ICs.columns = fac_name
        return ICs
    
    def multi_cumICs(self, fac_name:list=None,look_forward=None, method="spearman"):
        return self.multi_ICs(fac_name,look_forward,method).cumsum()
    
    def single_get_group_assets(self,fac:str,factor_quantile):
        cf = self._cleaned_factors[fac]
        tmp = cf[cf['factor_quantile'] == factor_quantile]["factor_quantile"]
        return tmp
    
    def single_get_group_positions(self,fac:str,factor_quantile: int):
        """某年因子的某组的等权组合权重"""
        cf = self._cleaned_factors[fac]
        tmp = cf[cf['factor_quantile'] == factor_quantile]["factor_quantile"].unstack().reindex(self.universe, axis=1)
        weights = (tmp > 0) / np.repeat((tmp>0).sum(axis=1).values.reshape(-1,1), self.n_assets, axis=1)
        return weights
    
    def single_get_long_short_positions(self,fac:str):
        """某年因子的某组的多空组合权重"""
        weights1 = self.single_get_group_positions(fac,factor_quantile=1)
        weights2 = self.single_get_group_positions(fac,factor_quantile=self._quantiles)
        return weights2/2-weights1/2
    
    def single_group_return(self,fac:str,factor_quantile:int,limit_up=True,limit_down=True):
        weights = self.get_group_positions(fac,factor_quantile)
        return self.backtest.run_backtest(weights,limit_up,limit_down)
    
    def single_groups_return(self,fac:str,limit_up=True,limit_down=True):
        """这里是returns,不是PnL,懒得改"""
        returns = []
        for i in range(1,self._quantiles + 1):
            rt = self.single_group_return(fac,factor_quantile=i,limit_up=limit_up,limit_down=limit_down)
            returns.append(rt)
        returns = pd.concat(returns,axis=1)
        returns.columns = [f'group{i}' for i in range(1,self._quantiles + 1)]
        return returns
    
    def multi_long_short_return(self,fac_name:list=None,limit_up=True,limit_down=True):
        returns = []
        if fac_name is None:
            fac_name = self._factor_name
        for fac in fac_name:
            weights = self.single_get_long_short_positions(fac)
            rt = self.backtest.run_backtest(weights,limit_up,limit_down)
            returns.append(rt)
        returns = pd.concat(returns,axis=1)
        returns.columns = fac_name
        return returns
    
    def single_hypo_testing(self,fac:str,look_forward=None,method='RLM',alpha=0.05):
        if self.periods_by_factor:
            look_forward = '1T'
        else:
            if look_forward is None:
                raise ValueError('look_forward should not be None if you do not set periods_by_factor.')
            look_forward = f"{look_forward}D"
        test_bool = []
        cf = self._cleaned_factors[fac]
        for date in cf.index.get_level_values(0).drop_duplicates():
            X = cf.loc[date]['factor']
            y = cf.loc[date][look_forward]
            test = StatTtest(X,y,method=method)
            test_bool.append(test.t_test(alpha=alpha))
        test_frame = DataFrame(np.array(test_bool))
        if len(test_frame.columns) != 3:
            print(test_frame) 
            print(cf)
        test_frame.columns = ['beta','u','d']
        total = len(test_frame.index)
        if total == 0:
            return Series(np.zeros(8),
                          index = ['正向显著比例','负向显著比例','正-负','同向显著次数比','状态切换次数比','同向-切换','因子方向','总检验次数'])
        u = test_frame['u'].sum()
        d = test_frame['d'].sum()
        pass_frame = test_frame[(test_frame['u']==True)|(test_frame['d']==True)]
        sep = pd.concat([pass_frame['beta'],pass_frame['u'].shift(1)],axis=1).dropna()
        sep.columns = ['0','1']
        same = (sep['0'] == sep['1']).sum()
        dif = (sep['0'] != sep['1']).sum()
        if u>=d:
            direct = 1
        else:
            direct = -1
        return Series([u/total,d/total,(u-d)/total, same/total,dif/total,(same-dif)/total,direct,total],
                      index = ['正向显著比例','负向显著比例','正-负','同向显著次数比','状态切换次数比','同向-切换','因子方向','总检验次数']) 

    def multi_hypo_testing(self,fac_name:list=None,look_forward=None,method='RLM',alpha=0.05):
        if fac_name is None:
            fac_name = self._factor_name
        results = []
        for fac in fac_name:
            ht = self.single_hypo_testing(fac,look_forward,method,alpha)
            results.append(ht)
        results = pd.concat(results,axis=1)
        results.columns = fac_name
        return results

    
