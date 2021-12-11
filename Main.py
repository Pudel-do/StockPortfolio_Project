
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 09:51:47 2021

@author: Matthias Pudel
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import scipy.optimize as sco
import os
import warnings
warnings.filterwarnings("ignore")

#==================================================================
from PortfolioConstruction import *
from PredictiveAnalytics import *
from ImportData import *
from Visualization import *

#==================================================================
fund_update = True
use_training = True
scaler = MinMaxScaler()
model = SVR()
param_grid = {'kernel': ('linear', 'rbf','poly'), 
              'C':[1.5, 10],
              'gamma': [1e-7, 1e-4],
              'epsilon':[0.1,0.2,0.5,0.3]}
#==================================================================
start_date = pd.Timestamp('2019-01-01')
stock_ticks = [
    'ALB', 'SQM', 
    'PLUG', 'BLDP', 
    ]
stock_ticks.sort()
bench_ticks = [
    '^GSPC', '^GDAXI', 
    'URTH', '^HSI', 'BTC-USD'
    ]
bench_ticks.sort()
investment = 2500
port_type = 'Maximum Sharpe Ratio'
benchmark = '^GDAXI'
pred_range = 1
model_vars = [
    'totalAssets', 'totalDebt',
    'totalLiabilities', 'cashAndShortTermInvestments',
    'longTermInvestments', 'shortTermDebt',
    'retainedEarnings', 'netDebt', 'revenuePerShare',
    'peRatio', 'priceToSalesRatio', 'debtToAssets',
    'returnOnAssets', 'netProfitMargin',
    'Price'
    ]
#==================================================================

quotes = GetQuotes(stock_ticks, start_date)
stock_rets = GetReturns(stock_ticks, start_date)
bench_rets = GetReturns(bench_ticks, start_date)

print(40*'=')
print('Predictive Analysis')
print(40*'=')
fundamentals = PreprocessFunds(stock_ticks, 
                                model_vars,
                                pred_range,
                                fund_update)
    
model_funds = fundamentals['Model']
raw_funds = fundamentals['Raw']

pred_analysis = PredictiveAnalysis(model_funds, 
                                  model_vars, 
                                  use_training)
    
feat_scaling = pred_analysis.feature_selection(scaler)
model = pred_analysis.model_construction(model, param_grid)
prediction = pred_analysis.target_prediction(raw_funds, quotes)

    
print(40*'=')
print('Technical Analysis')
print(40*'=')
port_analysis = PortfolioAnalysis(stock_ticks,
                                  stock_rets, 
                                  quotes, 
                                  investment,
                                  port_type)

returns = ReturnAggregation(stock_ticks,
                            stock_rets,
                            bench_rets,
                            port_analysis,
                            port_type)

Print_PortfolioResults(port_analysis, 
                        stock_rets,
                        bench_rets,
                        investment, 
                        port_type,
                        benchmark)

Print_ValueAtRisk(returns, investment)

Plot_MovingAverage(quotes, False)
Plot_PriceVolumeChart(quotes, True)
Plot_ReturnAnalysis(returns, True)
Plot_Portfolio_Correlation(returns, benchmark, False)




