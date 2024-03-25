import numpy as np
import pandas as pd
from pandas import Series,DataFrame
from typing import Iterable



class DataHandle:

    @staticmethod
    def handle_nan_value(c:DataFrame|Series,method='drop'):
        l = len(c)
        if method == 'drop':
            c = c.dropna()
            if len(c)/l < 0.5:
                print('You have too many None value, please check your data.')
        
        elif method == 'ffill':
            c = c.ffill().dropna()

        elif method == 'bfill':
            c = c.bfill().dropna()
        
        elif method == 'mfill':
            c = c.fillna(c.mean())
        
        elif method == 'fill0':
            c = c.fillna(0)
        
        else:
            raise ValueError('method should be drop, ffill, bfill, mfill, fill0.')
        
        return [c[col] for col in c.columns]

    @staticmethod
    def handle_nan_value_series(data:Iterable[Series],method='drop'):
        """
        This function is to handle the nan value and align index for two Series data with similar index, you can pick the different methed to deal with.

        Params:
        - data:Iterable[Series], you should input two Seriers at least.

        - method:
            drop: drop the nan value.
            ffill: ffill the nan value.
            bfill: bfill the nan value.
            mfill: use the mean value to fill the nan value.
            fill0: use 0 to fill the nan value.
        
        - Return: Iterable[Series]
        """
        c:DataFrame = pd.concat(data,axis=1)
        l = len(c)
        if method == 'drop':
            c = c.dropna()
            if len(c)/l < 0.5:
                print('You have too many None value, please check your data.')
        
        elif method == 'ffill':
            c = c.ffill().dropna()

        elif method == 'bfill':
            c = c.bfill().dropna()
        
        elif method == 'mfill':
            c = c.fillna(c.mean())
        
        elif method == 'fill0':
            c = c.fillna(0)
        
        else:
            raise ValueError('method should be drop, ffill, bfill, mfill, fill0.')
        
        return [c[col] for col in c.columns]
        

    @staticmethod
    def handle_nan_value_X_y(X:DataFrame,y:Series,method='drop'):
        """
        This function is to handle the nan value and align index for X and y, you can pick the different methed to deal with.

        Params:
        - X:DataFrame
        - y:Series

        - method: pick a method to deal with the X data.
            drop: drop the nan value.
            ffill: ffill the nan value.
            bfill: bfill the nan value.
            mfill: use the mean value to fill the nan value.
            fill0: use 0 to fill the nan value.
        
        Return: (X_,y_)

        Warning: use 'y' as the columns name, you should ensure 'y' is not in X.columns.
        """

        if 'y' in X.columns:
            raise ValueError(' str(y) should not in X.columns')
        
        l = len(X)

        if method == 'drop':
            X_ = X.dropna().copy()
            if len(X_)/l < 0.5:
                print('You have too many None value, please check your data.')
        
        elif method == 'ffill':
            X_ = X.ffill().dropna().copy()

        elif method == 'bfill':
            X_ = X.bfill().dropna().copy()
        
        elif method == 'mfill':
            X_ = X.fillna(X.mean()).copy()
        
        elif method == 'fill0':
            X_ = X.fillna(0).copy()
        
        else:
            raise ValueError('method should be drop, ffill, bfill, mfill, fill0.')
        
        y_ = y.copy()
        y_.name = 'y'
        data = pd.concat([y_,X_],axis=1).dropna()
        y_ = data['y']
        X_ = data.drop(columns=['y'])
        
        return (X_,y_)
    
    @staticmethod
    def meanize(data:DataFrame|Series):
        return data - data.mean(numeric_only=True)
    
    @staticmethod
    def normalize(data:DataFrame|Series):
        return (data - data.mean(numeric_only=True))/data.std(numeric_only=True)
    
    @staticmethod
    def shrink_extreme_value(data: pd.DataFrame | pd.Series):
        rate = 0.05
        lower = data.quantile(rate)
        upper = data.quantile(1 - rate)

        if isinstance(data, pd.DataFrame):
            axis_value = 1
        elif isinstance(data, pd.Series):
            axis_value = 0
        else:
            raise ValueError("Unsupported data type. Use DataFrame or Series.")

        return data.clip(lower=lower, upper=upper, axis=axis_value)
    
    @staticmethod
    def normalize_and_shrink(data:DataFrame|Series):
        nor = DataHandle.normalize(data)
        nor_shr = DataHandle.shrink_extreme_value(nor)
        return nor_shr.values
    
    @staticmethod
    def multi_index_meanize(data:DataFrame,level=[0]):
        return data.groupby(level=level).transform(DataHandle.meanize)
    
    @staticmethod
    def multi_index_normalize(data:DataFrame,level=[0]):
        return data.groupby(level=level).transform(DataHandle.normalize)
    
    @staticmethod
    def multi_index_shrink(data:DataFrame,level=[0]):
        return data.groupby(level=level).transform(DataHandle.shrink_extreme_value)
    
    @staticmethod
    def group_normalize(data:DataFrame, group:dict):
        return None
        mean = data.groupby(by=group).mean(numeric_only=True)
        std = data.groupby(by=group).std(numeric_only=True)
        return (data-mean)/std