# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 09:52:26 2021

@author: Matthias Pudel
"""

import pandas as pd
import numpy as np
import yfinance as yf
import FundamentalAnalysis as fa
import pickle
import os

import warnings
warnings.filterwarnings("ignore")

path = r'C:\Users\Matthias Pudel\OneDrive\Private Finance\Portfolio'
file_api_key = 'api_key.txt'
file = open(os.path.join(path, file_api_key),'r')  
api_key = file.read()

#================================================

# save_obj(quotes, path, file_quotes)
# stock_rets.to_csv(path + '\\' + file_stock_rets)
# bench_rets.to_csv(path + '\\' + file_bech_rets)


def save_obj(obj, path, filename):
    with open(path + '\\' + filename, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def GetFile(path, filename):
    with open(path + '\\' + filename, 'rb') as f:
        return pickle.load(f)
    

def GetReturns_Offline(path, filename):
    file = pd.read_csv(path + '\\' + filename, index_col = 'Date')
    file.index = pd.to_datetime(file.index)
    return file
    


def GetQuotes(ticks, start_date):
    quotes = {}
    for tick in ticks:
        data = yf.download(tick, 
                           start = start_date, 
                           threads = True,
                           progress = False)
        quotes[tick] = data
    return quotes

           
def GetReturns(ticks, start_date):
    data = yf.download(ticks, 
                       start = start_date, 
                       threads = True,
                       progress = False)
    data = data['Adj Close']
    data = data.fillna(method = 'bfill')
    
    returns = np.log(data / data.shift(1))
    returns = returns.dropna()
    
    return returns


def GetFundamentals(ticks, path, file_funds):
    period = 'annual'
    fund_data = {}
    for tick in ticks:
        balance_sheet = fa.balance_sheet_statement(tick, 
                                                   api_key, 
                                                   period = period)
        
        balance_sheet = balance_sheet.transpose()
        
        key_metrics = fa.key_metrics(tick, 
                                     api_key, 
                                     period = period)
        key_metrics = key_metrics.transpose()
        
        com_columns = balance_sheet.columns.intersection(key_metrics.columns)
        key_metrics = key_metrics.drop(columns = com_columns)
        funds = balance_sheet.join(key_metrics, how = 'left')
        
        ratios = fa.financial_ratios(tick, 
                                     api_key, 
                                     period = period)
        ratios = ratios.transpose()
        
        com_columns = funds.columns.intersection(ratios.columns)
        ratios = ratios.drop(columns = com_columns)
        funds = funds.join(ratios, how = 'left')
 
        fund_data[tick] = funds
    
    save_obj(fund_data, 
             path, 
             file_funds)
    
    return fund_data
        
        
        
        
        
        




    
    











        