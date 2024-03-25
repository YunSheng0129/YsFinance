import numpy as np
import pandas as pd
from pandas import Series
from cmath import exp
from math import pi, sqrt
from functools import partial


def stat_returns(returns:Series):
    means = returns.mean()*252
    std = returns.std()*sqrt(252)
    SR = (means-0.02)/std
    return Series([means,std,SR],index=['mean','std','SR'])
    


class Filter:
    
    @staticmethod
    def SMA(data:Series,window):
        return data.rolling(window=window,min_periods=1).mean()
    
    @staticmethod
    def EMA(data:Series,alpha):
        return data.ewm(alpha=alpha).mean()
    
    @staticmethod
    def WMA(data:Series, weights:np.array):
        weights = weights/weights.sum()
        def func(x):
            try:
                return (x*weights).sum()
            except ValueError:
                return np.nan
        window = len(weights)
        return data.rolling(window=window).apply(func)
    
    @staticmethod
    def VI(data:Series, k):
        den = np.abs(data - data.shift(1)).rolling(window=k).sum()
        num = np.abs(data - data.shift(k))
        return num/den
    
    @staticmethod
    def VMA(data:Series,k,alpha):
        VI = Filter.VI(data,k)
        vma = Series(np.nan,index=data.index)
        vma.iloc[k-1] = data.iloc[0:k].mean() 
        for i in range(k,len(data)):
            vma.iloc[i] = (alpha * VI.iloc[i])*data.iloc[i] + (1-alpha*VI.iloc[i])*vma.iloc[i-1]
        return vma

    @staticmethod
    def DEMA(data:Series,alpha):
        ema1 = Filter.EMA(data,alpha)
        ema2 = Filter.EMA(ema1,alpha)
        return 2*ema1-ema2
    
    @staticmethod
    def TRIX(data:Series,alpha):
        ema = Filter.EMA(data,alpha)
        ema = Filter.EMA(ema,alpha)
        ema = Filter.EMA(ema,alpha)
        return ema
    
    @staticmethod
    def HMA(data:Series,k):
        weights1 = k/2-np.flip(np.arange(k/2))
        weights2 = k - np.flip(np.arange(k))
        weights3 = sqrt(k)-np.flip(np.arange(sqrt(k)))
        wma1 = Filter.WMA(data,weights1)
        wma2 = Filter.WMA(data,weights2)
        wma3 = Filter.WMA(2*wma1-wma2,weights3)
        return wma3
    
    @staticmethod
    def MACD(data:Series,k,k1,k2):
        if k1 >= k2:
            raise ValueError('k1 < k2 is available.')
        sma1 = Filter.SMA(data,k1)
        sma2 = Filter.SMA(data,k2)
        sma3 = Filter.SMA(sma1-sma2,k)
        return sma3
    
    @staticmethod
    def moving_std(data:Series,window:int):
        return data.rolling(window=window,min_periods=2).std()
    
    @staticmethod
    def Bollinger_bands(data:Series,k=int,m=2):
        sigma = Filter.moving_std(data,k)
        middle_rail = Filter.SMA(data,k)
        upper_rail = middle_rail + m*sigma
        lower_rail = middle_rail - m*sigma
        z = (data-middle_rail)/sigma
        return (z,middle_rail,upper_rail,lower_rail)
    
    @staticmethod
    def RSI(data:Series,k:int):
        def func_postive(a):
            if a > 0:
                return a
            return 0
        num = Filter.SMA((data-data.shift(1)).apply(func_postive),window=k)
        den = Filter.SMA(np.abs(data-data.shift(1)),window=k)
        result = (num/den)
        result.iloc[:k] = np.nan
        return result
    
    @staticmethod
    def KDJ(data:Series,k:int,alpha):
        H = data.rolling(window=k,min_periods=2).max()
        L = data.rolling(window=k,min_periods=2).min()
        K = Filter.EMA((data-L)/(H-L),alpha=alpha)
        D = Filter.EMA(K,alpha)
        J = 3*K-2*D
        return (K,D,J,H,L)
    
    @staticmethod
    def HP(data:Series,lam,k):
        def HP_range(data:Series):
            if data.isna().sum() > 0:
                raise ValueError('data have none type, please drop them or fill them!')
            def matrix_D(n):
                if n < 3:
                    raise ValueError(f'matrix_D dimision is too small with n = {n}')
                return np.eye(n-2,n) - 2*np.eye(n-2,n,1) + np.eye(n-2,n,2)
            n = len(data)
            A = np.linalg.inv(np.eye(n) + lam * matrix_D(n).T @ matrix_D(n))
            return( A @ data.values)[-1]
        return data.rolling(window=k).apply(HP_range)
        
        

    @staticmethod
    def L1(data:Series,window,lam):
        pass

    @staticmethod
    def Fourier(data:Series,k):
        po = data
        ne = np.flip(data).shift(1).dropna()
        x = pd.concat([ne,po])
        x.index = np.arange(-len(data)+1,len(data),dtype=int)
        n = int(len(x)/2)
        def y(j:int, x):
            y_ = x.loc[0]
            for k in range(1,n+1):
                y_ += x.loc[k] * exp(2*pi*complex(imag=1)*k*j/(2*n+1))
                y_ += x.loc[-k] * exp(2*pi*complex(imag=1)*k*j/(2*n+1))
            return y_.real
        
        new_y = Series(list(map(partial(y,x=x),x.index)),index=x.index)
        new_y.loc[np.abs(new_y.index) > k] = 0
        new_x = Series(list(map(partial(y,x=new_y),x.index)),index=x.index)/(2*n+1)
        new_x = new_x.loc[0:]
        new_x.index = data.index
        return new_x
    
    @staticmethod
    def Kalman(data:Series,u,v,p0=0.5):
        p = [p0]
        q = [0]
        y = data.values
        new_x = [y[0]]
        for n in range(1,len(data)):
            q.append(p[n-1]+u**2)
            lam = v**2/(q[n]+v**2)
            new_x.append(lam*new_x[n-1] + (1-lam)*y[n-1])
            p.append(q[n]*v**2/(q[n]+v**2))
        return Series(new_x,index=data.index)



class MoveSignal:

    @staticmethod
    def MACD_move_signal(short_term_filter:Series,long_term_filter:Series,k:int=1,gap=0.05):
        MACD = Filter.SMA(short_term_filter-long_term_filter,window=k)
        signal = Series(index=MACD.index)
        signal.loc[MACD > gap] = 1
        signal.loc[MACD < -gap] = -1
        signal.loc[(MACD <= gap) & (MACD >= -gap)] = 0
        return signal
    
    @staticmethod
    def pure_MACD_move_signal(data:Series,k1,k2,k=1,gap=0.05):
        MACD = Filter.MACD(data,k,k1,k2)
        signal = Series(index=MACD.index)
        signal.loc[MACD > gap] = 1
        signal.loc[MACD < -gap] = -1
        signal.loc[(MACD <= gap) & (MACD >= -gap)] = 0
        return signal

    
    @staticmethod
    def reverse_move_signal(reverse:Series,gap):
        signal = Series(index=reverse.index)
        signal.loc[reverse>gap] = -1
        signal.loc[reverse<-gap] = 1
        signal.loc[(reverse<=gap)&(reverse>=-gap)] = 0
        return signal
    
    @staticmethod
    def signal_runback(price:Series,signal:Series,short=True,fee=0.0002):
        signal = signal.copy()
        if not short:
            signal.loc[signal<0] = 0
        index = signal.index[0]
        returns = [-fee]
        for k in signal.index[1:]:
            returns.append((price.loc[k]/price.loc[index]-1)*signal.loc[index]-fee*np.abs(signal.loc[k]-signal.loc[index]))
            index = k
        return Series(returns, index=signal.index)
    
    @staticmethod
    def hold_statistics(signal):
        short_num = (signal>0).sum()
        long_num = (signal<0).sum()
        no_num = (signal==0).sum()
        all_num = short_num + long_num + no_num
        change_num = (np.abs(signal-signal.shift(1))>0.01).sum()
        return Series([short_num/all_num,long_num/all_num,no_num/all_num,change_num/all_num,all_num],index=['short','long','none','change','all'])












