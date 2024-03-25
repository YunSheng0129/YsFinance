import numpy as np
import pandas as pd
from pandas import Series,DataFrame
from .tools import DataHandle
import statsmodels.api as sm
import scipy.stats as scist


class StatReturns:
    """
    Input the returns series with freq-D, and get the statistics about the returns. If you input the benchmark returns series, you can get the alpha,beta and so on.

    Params: 
    - returns:DataFrame. Give the returns dataframe with day-freq
    - benchmark:Series. Give the benchmark asset returns series with day-freq. If you don't provide the data, you will not get the correlative analysis results.
    - start_date=None. The start date you analysis, if you do not provide, start date will be the returns data's start.
    - end_date=None. The end date you analysis, if you do not provide, end date will be the returns data's end.

    """
    def __init__(self,returns:DataFrame,benchmark:Series=None,start_date=None,end_date=None):
        if start_date is None:
            self._start_date = returns.index[0]
        else:
            self._start_date = start_date
        if end_date is None:
            self._end_date = returns.index[-1]
        else:
            self._end_date = end_date
        
        self._returns = returns.loc[start_date:end_date].copy()
        self._pnl = (1+self._returns).cumprod()
        if benchmark is not None:
            self._benchmark = benchmark.loc[start_date:end_date]
        else:
            self._benchmark = None


    def stat_frame(self,market:bool=False,window=90,q=0.05,rf=0):
        """
        Params:
        - market:bool=False. If market bool is True, you will get the market-correlative analysis, but ensure you provide the benchmark data.
        - window=90. Natual day number when you calculate the max drawdown, and it mutiply 21/30 as the trade day number.
        - q=0.05. VaR quantile. 
        - rf=0. Annualized risk free rate, this will be used when you calculate the sharpe ratio and linear regression.

        Results:Dataframe. With asset index and analysis project columns.

        Attentions:
        - 252 trading days a year.

        """
        means = self._returns.mean()*252
        stds = self._returns.std()*np.sqrt(252)
        sharpe_ratio = (means-rf)/stds
        max_drawdown = (1-self._pnl/(self._pnl.rolling(window=int(window*21/30)).max())).max() #每30天约为21个交易日，window为自然日
        VaR = self._returns.quantile(q)

        if not market:
            result = pd.concat([means,stds,sharpe_ratio,max_drawdown,VaR],axis=1)
            result.columns = ['mean','std','SR',f'mdd_{window}D',f'VaR_{q}']
            return result
        
        if self._benchmark is None:
            raise ValueError('You have not provide the benchmark returns.')
        
        alpha = []
        beta = []
        residual_std = []
        X,y = DataHandle.handle_nan_value_X_y(self._returns,self._benchmark,method='mfill')
        for col in self._returns.columns:
            params = np.polyfit(X[col],y,deg=1)
            be,al = params
            res = np.std(y-np.polyval(params,X[col]))
            beta.append(be)
            alpha.append(al*252)
            residual_std.append(res)
        beta = Series(beta,index = X.columns)
        alpha = Series(alpha, index = X.columns)-rf
        residual_std = Series(residual_std,index=X.columns)
        TR = (means-rf)/beta
        IR = alpha/residual_std/np.sqrt(252)
        result = pd.concat([means,stds,sharpe_ratio,max_drawdown,VaR,beta,alpha,TR,IR],axis=1)
        result.columns = ['mean','std','SR',f'mdd_{window}D',f'VaR_{q}','beta','alpha','TR','IR']
        return result


def stat_returns(returns,benchmark=None,start_date=None,end_date=None,market=False,window=90,q=0.05,rf=0):
    stat = StatReturns(returns=returns,benchmark=benchmark,start_date=start_date,end_date=end_date)
    frame = stat.stat_frame(market=market,window=window,q=q,rf=rf)
    return frame




class StatTtest:
    def __init__(self,X:Series,y:Series,method):
        data = pd.concat([X,y],axis=1).dropna()
        data.columns = ['X','y']
        self.data = data
        self.X = data['X']
        self.y = data['y']
        self.degree = len(self.data.index)
        self.method = method
    
    def reg_coef(self):
        if len(self.X) != len(self.y):
            raise ValueError("X,y should have the same dimension")
        if self.method == 'OLS':
            X = sm.add_constant(self.data['X'].values)
            y = self.data['y'].values
            model = sm.OLS(y,X).fit()
            beta = model.params[1]
            sr = model.bse[1]
            return (beta, beta/sr)
        if self.method == 'RLM':
            X = sm.add_constant(self.data['X'].values)
            y = self.data['y'].values
            model = sm.RLM(y,X).fit()
            beta = model.params[1]
            sr = model.bse[1]
            return (beta, beta/sr)
        if self.method == 'QR':
            X = sm.add_constant(self.data['X'].values)
            y = self.data['y'].values
            model = sm.QuantReg(y,X).fit(max_iter=3000)
            beta = model.params[1]
            sr = model.bse[1]
            return (beta, beta/sr)
        raise ValueError("method should be OLS, RLM or QR")
    
    def t_test(self,alpha=0.05):
        """返回系数是否大于0，是否显著大于0和小于0的bool值列表"""
        beta, t_value = self.reg_coef()
        bool_lst = [beta>0]
        if t_value > scist.t.ppf(1-alpha,df=self.degree-2):
            bool_lst.append(True)
        else:
            bool_lst.append(False)
        if t_value < scist.t.ppf(alpha,df=self.degree-2):
            bool_lst.append(True)
        else:
            bool_lst.append(False)
        return bool_lst
  