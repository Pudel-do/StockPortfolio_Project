# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 19:46:58 2021

@author: Matthias Pudel
"""

import pandas as pd
import numpy as np
import scipy.optimize as sco
import scipy.stats as scs

#================================================

from ImportData import GetReturns

def port_ret(weights, returns):
    return np.sum(returns.mean() * weights) * 252

def port_vol(weights, returns):
    return np.sqrt(np.dot(weights.T, 
                          np.dot(returns.cov() * 252, 
                                 weights)))

def sharpe_ratio(weights, returns):
    return -port_ret(weights, returns) / port_vol(weights, returns)


def max_sharpe_weights(ticks, returns):
    noa = len(ticks)
    constraints = ({'type': 'eq',
                    'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0,1) for x in range(noa))
    eq_weights = np.array(noa * [1 / noa,])
    
    opt = sco.minimize(sharpe_ratio,
                       eq_weights,
                       returns,
                       method = 'SLSQP',
                       bounds = bounds,
                       constraints = constraints)
        
    weights = opt['x']
    
    output = {'Stock': ticks,
              'Weight': weights}
    output = pd.DataFrame(output)
    
    return output


def min_var_weights(ticks, returns):
    noa = len(ticks)
    constraints = ({'type': 'eq',
                    'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0,1) for x in range(noa))
    eq_weights = np.array(noa * [1 / noa,])
    
    opt = sco.minimize(port_vol,
                       eq_weights,
                       returns,
                       method = 'SLSQP',
                       bounds = bounds,
                       constraints = constraints)
        
    weights = opt['x']
    
    output = {'Stock': ticks,
              'Weight': weights}
    output = pd.DataFrame(output)
    
    return output


def ReturnAggregation(ticks, stock_rets, bench_rets, port_analysis, port_type):
    if port_type == 'Maximum Sharpe Ratio':
        opt_weights = max_sharpe_weights(ticks, stock_rets)
    else:
        opt_weights = min_var_weights(ticks, stock_rets)
        
    noa = len(ticks)    
    opt_weights = np.array(opt_weights['Weight'])
    eq_weights = np.array(noa * [1 / noa,])
    
    opt_rets = (stock_rets * opt_weights).sum(axis = 1)
    opt_rets = pd.DataFrame(opt_rets)
    opt_rets.columns = ['Optimal_Portfolio']
    
    eq_rets = (stock_rets * eq_weights).sum(axis = 1)
    eq_rets = pd.DataFrame(eq_rets)
    eq_rets.columns = ['Equal_Weighted_Portfolio']

    port_weights = np.array(port_analysis['Portfolio Weights'])
    port_rets = (stock_rets * port_weights).sum(axis = 1)
    port_rets = pd.DataFrame(port_rets)
    port_rets.columns = ['Portfolio']
    
    rets = bench_rets.join([opt_rets, eq_rets, port_rets], 
                           how = 'left')
    rets = rets.dropna()
    
    return rets

        
    
def PortfolioAnalysis(ticks, stock_rets, quotes, investment, port_type):
    if port_type == 'Maximum Sharpe Ratio':
        opt_weights = max_sharpe_weights(ticks, stock_rets)
    else:
        opt_weights = min_var_weights(ticks, stock_rets)
        
    opt_weights = np.array(opt_weights['Weight'])
    
    last_quotes = []
    
    for key, value in quotes.items():
        last_quote = value['Adj Close'].values[-1]
        last_quotes.append(last_quote)
        
    last_quotes = np.array(last_quotes)
    
    stock_invest = opt_weights * investment
    long_pos = stock_invest / last_quotes
    long_pos = np.around(long_pos).astype(int)

    port_weights = long_pos * last_quotes
    port_weights = port_weights / investment
    
    results = pd.DataFrame({'Stock': ticks,
                            'Long Position': long_pos,
                            'Price': last_quotes,
                            'Portfolio Weights': port_weights,
                            'Optimal Weights': opt_weights})
    
    return results


def ValueAtRisk(returns, investment):
    port_rets = returns['Portfolio']
    percs = np.array([1.0, 2.5, 5.0, 10.0])
    loss = scs.scoreatpercentile(investment * port_rets, 
                                 percs)
    conf_lvl = 100 - percs

    result = pd.DataFrame({'Confidence Level': conf_lvl,
                           'Loss': np.absolute(loss)})
    
    return result



