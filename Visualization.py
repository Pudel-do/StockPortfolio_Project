# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 10:52:07 2021

@author: Matthias Pudel
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import scipy.optimize as sco
import os
from tabulate import tabulate
import matplotlib.dates as mpl_dates

import warnings
warnings.filterwarnings("ignore")

#================================================

from PortfolioConstruction import *


xlabel_size = {'fontsize' : 15}
ylabel_size = {'fontsize' : 15}
suptitle_size = 25
title_size = {'fontsize' : 20}
legend_size = {'size': 20}


def Print_PortfolioResults(results, stock_rets, bench_rets, investment, port_type, benchmark):
    port_weights = results['Portfolio Weights']
    port_weights = np.array(port_weights)
    eq_weights = np.array(len(results) * [1 / len(results),])
    
    anual_ret = port_ret(port_weights, stock_rets)
    anual_vol = port_vol(port_weights, stock_rets)
    sharpe = sharpe_ratio(port_weights, stock_rets)
    sharpe = np.absolute(sharpe)
    
    eq_anual_ret = port_ret(eq_weights, stock_rets)
    eq_anual_vol = port_vol(eq_weights, stock_rets)
    eq_sharpe = sharpe_ratio(eq_weights, stock_rets)
    eq_sharpe = np.absolute(sharpe)
    
    benchmark_rets = bench_rets[benchmark]
    bench_anual_ret = benchmark_rets.mean() * 252
    bench_anual_vol = benchmark_rets.std() * np.sqrt(252)
    bench_sharpe = bench_anual_ret / bench_anual_vol
    
    sum_weights = np.sum(port_weights)
    port_costs = sum_weights * investment
    
    if benchmark == '^GSPC':
        benchmark_name = 'S&P 500'
    elif benchmark == '^GDAXI':
        benchmark_name = 'DAX'
    elif benchmark == 'URTH':
        benchmark_name = 'MSCI World'
    elif benchmark == '^HSI':
        benchmark_name = 'Hang Seng'
    elif benchmark == 'BTC-USD':
        benchmark_name = 'Bitcoin'
        
    idx = ['Annualized Return',
           'Annualized Volatility',
           'Sharpe Ratio']
    
    performance_comp = pd.DataFrame(index = idx)
    
    performance_comp.loc['Annualized Return', 'Portfolio'] = anual_ret
    performance_comp.loc['Annualized Volatility', 'Portfolio'] = anual_vol
    performance_comp.loc['Sharpe Ratio', 'Portfolio'] = sharpe
    
    performance_comp.loc['Annualized Return', 'Equal Weighted Portfolio'] = eq_anual_ret
    performance_comp.loc['Annualized Volatility', 'Equal Weighted Portfolio'] = eq_anual_vol
    performance_comp.loc['Sharpe Ratio', 'Equal Weighted Portfolio'] = eq_sharpe
    
    performance_comp.loc['Annualized Return', benchmark_name] = bench_anual_ret
    performance_comp.loc['Annualized Volatility', benchmark_name] = bench_anual_vol
    performance_comp.loc['Sharpe Ratio', benchmark_name] = bench_sharpe
    
    performance_comp = performance_comp.astype(float)
    
    print('\n')
    print(f'Portfolio Type: {port_type}')
    print(f'Investment Amount: {investment}')
    print(f'Portfolio Costs: {port_costs :.2f}')
    print(f'Sum of Portfolio Weights: {sum_weights :.2f}')
    
    print('\n')
    print(tabulate(performance_comp,
                   headers = 'keys',
                   showindex = True,
                   tablefmt = 'simple',
                   numalign = 'center',
                   floatfmt = ('.2f')))
    
    
    print('\n')
    print(tabulate(results,
                   headers = 'keys',
                   showindex = False,
                   tablefmt = 'simple',
                   numalign = 'center',
                   floatfmt = ('.2f')))
    
    
def Print_ValueAtRisk(returns, investment):
    var = ValueAtRisk(returns, investment)
    
    print('\n\n',
          tabulate(var,
                   headers = 'keys',
                   showindex = False,
                   tablefmt = 'simple',
                   numalign = 'left',
                   floatfmt = ('.2f')))
    
    

def Plot_ReturnAnalysis(returns, plotting):
    if plotting == True:
        plt.rc('figure', figsize = (20,10))
        plt.plot(returns['Optimal_Portfolio'].cumsum().apply(np.exp), 
                 lw = 1.5,
                 label = 'Optimal Portfolio')
        plt.plot(returns['Equal_Weighted_Portfolio'].cumsum().apply(np.exp),
                 lw = 1.5,
                 label = 'Equal Weighted Portfolio')
        plt.plot(returns['Portfolio'].cumsum().apply(np.exp),
                 lw = 1.5,
                 label = 'Portfolio')
        plt.plot(returns['^GSPC'].cumsum().apply(np.exp),
                 'b', lw = 1.5,
                 linestyle = 'dashed',
                 label = 'SP500')
        plt.plot(returns['^GDAXI'].cumsum().apply(np.exp),
                 'g', lw = 1.5,
                 linestyle = 'dashed',
                 label = 'DAX')
        plt.plot(returns['URTH'].cumsum().apply(np.exp),
                 'r', lw = 1.5,
                 linestyle = 'dashed',
                 label = 'MSCI World')
        plt.plot(returns['^HSI'].cumsum().apply(np.exp),
                 'k', lw = 1.5,
                 linestyle = 'dashed',
                 label = 'Hang Seng')
        plt.legend(loc = 0, prop = legend_size)
        plt.grid(True)
        plt.xlabel('Date', fontdict = xlabel_size)
        plt.ylabel('Cumulative Returns', fontdict = ylabel_size)
        plt.title('Cumulative Returns', 
                  fontdict = title_size)
        plt.show()
    

    
    
def Plot_MovingAverage(quotes, plotting):
    if plotting == True:
        ma_low = 42
        ma_high = 252
    
        for key, value in quotes.items():
            moving_avg = value['Adj Close']
            moving_avg = pd.DataFrame(moving_avg)
            
            moving_avg_low = moving_avg.rolling(ma_low).mean()
            moving_avg_low.columns = ['MA_Low']
            moving_avg_high = moving_avg.rolling(ma_high).mean()
            moving_avg_high.columns = ['MA_High']
            
            moving_avg = moving_avg.join([moving_avg_low, moving_avg_high],
                                         how = 'left')
            moving_avg.dropna(inplace = True)
            moving_avg['Position'] = np.where(moving_avg['MA_Low'] > moving_avg['MA_High'], 1, -1)
            
            fig, ax1 = plt.subplots(figsize = (20,10))
            plt.plot(moving_avg['Adj Close'],
                     'b', lw = 1.5, label = 'Adjusted Close Price')
            plt.plot(moving_avg['MA_Low'],
                     'g', lw = 1.25, label = f'Moving Average {ma_low} Days')
            plt.plot(moving_avg['MA_High'],
                     'y', lw = 1.25, label = f'Moving Average {ma_high} Days')
            plt.legend(loc = 0, prop = legend_size)
            plt.grid(True)
            plt.xlabel('Date', fontdict = xlabel_size)
            plt.ylabel('Stock Price', fontdict = ylabel_size)
            plt.title(key, fontdict = title_size)
            
            ax2 = ax1.twinx()
            plt.plot(moving_avg['Position'],
                     'r', lw = 0.75, 
                     linestyle = 'dashed',
                     label = 'Buy and Sell')
            plt.grid(False)
            plt.xlabel('Date', fontdict = xlabel_size)
            plt.ylabel('Buy and Sell', fontdict = ylabel_size)
            plt.title(key, fontdict = title_size)
            plt.show()

            
    
def Plot_PriceVolumeChart(quotes, plotting):
    if plotting == True:
        for key, value in quotes.items():
            plt.rc('figure', figsize=(20, 10))
            fig, ax = plt.subplots(2,1,
                                   gridspec_kw={'height_ratios': [3, 1]})
            fig.tight_layout(pad=3)
            date = value.index
            close = value['Adj Close']
            volume = value['Volume']
            
            plot_price = ax[0]
            plot_price.plot(date, close, 
                            'b', lw = 1.5, label = 'Price')
            plot_price.grid(True)
            plot_price.set_ylabel('Stock Price', fontdict = ylabel_size)
            plot_price.set_title(key, fontdict = title_size)
            
            plot_vol = ax[1]
            plot_vol.bar(date, volume, 
                         width=15, color='darkgrey')
            plot_vol.grid(False)
            plot_vol.set_ylabel('Trading Volume', fontdict = ylabel_size)
            plt.show()
        
            
            

def Plot_Portfolio_Correlation(returns, benchmark, plotting):
    if plotting == True:
        port_corr = returns['Portfolio'].rolling(window = 252).corr(returns[benchmark])
        port_corr_static = returns['Portfolio'].corr(returns[benchmark])
        
        if benchmark == '^GSPC':
            benchmark_name = 'S&P 500'
        elif benchmark == '^GDAXI':
            benchmark_name = 'DAX'
        elif benchmark == 'URTH':
            benchmark_name = 'MSCI World'
        elif benchmark == '^HSI':
            benchmark_name = 'Hang Seng'
    
        plt.figure(figsize = (20,10))
        plt.plot(port_corr,
                 'b',
                 lw = 1.5,
                 label = f'Correlation: {port_corr_static :.2}')
        plt.legend(loc = 0, prop = legend_size)
        plt.grid(True)
        plt.xlabel('Date', fontdict = xlabel_size)
        plt.ylabel('Dynamic Correlation', fontdict = ylabel_size)
        plt.title(f'Portfolio Correlation with {benchmark_name} Returns',
                  fontdict = title_size)
        plt.show()




    
    
