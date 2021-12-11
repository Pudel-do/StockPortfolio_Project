# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 14:55:08 2021

@author: Matthias Pudel
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import FundamentalAnalysis as fa
from sklearn.linear_model import Lasso
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from tabulate import tabulate
import os

import warnings
warnings.filterwarnings("ignore")

path = r'C:\Users\Matthias Pudel\OneDrive\Private Finance\Portfolio'
file_funds = 'Fundamentals.pkl'
file_params = 'ModelParameters.pkl'
file_best_est = 'ModelBestEstimator.pkl'

xlabel_size = {'fontsize' : 15}
ylabel_size = {'fontsize' : 15}
suptitle_size = 25
title_size = {'fontsize' : 20}
legend_size = {'size': 20}

#================================================

from ImportData import *


def PreprocessFunds(ticks, model_vars, pred_range, fund_update):
    if fund_update == True:
        fund_data = GetFundamentals(ticks, path, file_funds)
    else:
        fund_data = GetFile(path, file_funds)
    
    funds = {}
    funds['Raw'] = {}
    funds['Model']  = {}
    for key, value in fund_data.items():
        value['fillingDate'] = pd.to_datetime(value['fillingDate'])
        value['DateMerge'] = value['fillingDate'] - pd.tseries.offsets.BusinessDay(1)
        
        start_date = value['DateMerge'].min()
        end_date = pd.to_datetime('today').normalize() 
        quotes = yf.download(key, 
                             start = start_date, 
                             end = end_date,
                             threads = True,
                             progress = False)
        
        close_price = pd.DataFrame(quotes['Adj Close'])
        close_price.columns = ['PriceMerge']
        close_price.reset_index(level=0, inplace=True)
        
        value.reset_index(level=0, inplace=True)
        value = value.merge(close_price, 
                            left_on='DateMerge', 
                            right_on='Date',
                            how = 'left',
                            sort = True)
        value = value.drop(columns = ['Date'])
        value = value.rename(columns = {'index': 'Date'})
        value['Date'] = pd.to_datetime(value['Date'])
        value = value.set_index('Date')
        value = value.sort_index()
        value['Price'] = value['PriceMerge'].shift(-pred_range)
        
        cols = value.columns.tolist()
        cols = cols[-3:] + cols[:-3]
        value = value[cols]
        funds['Raw'][key] = value
        
        value = value[model_vars]
        value = value.astype(float)
        funds['Model'][key] = value
        
    return funds


    
    
class PredictiveAnalysis:
    
    def __init__(self, fund_data, model_vars, use_training):
        self.data = fund_data
        self.reg_vars = model_vars
        self.reg_vars.remove('Price')
        self.use_training = use_training
        
    def feature_selection(self, Scaler):
        regularization = Lasso(alpha = 0.1)
        self.model_data = {}
        self.pred_data = {}
        for key, value in self.data.items():
            regs = value[self.reg_vars]
            regs = regs.dropna()
            target = value['Price']
            
            scaler = Scaler
            scaling = scaler.fit(regs)
            regs_scaled = scaling.transform(regs)
            regs_scaled = pd.DataFrame(regs_scaled)
            regs_scaled.columns = regs.columns
            regs_scaled.index = regs.index
            
            model = regs_scaled.iloc[:-1, :]
            model = model.join(target, how = 'left')
            model = model.dropna()
            pred = regs_scaled.iloc[-1, :]
            pred = pd.DataFrame(pred)
            pred = pred.transpose()
                  
            coef = regularization.fit(model[self.reg_vars], model['Price']).coef_
            coef = pd.DataFrame(coef)
            coef = coef.transpose()
            coef = coef.abs()
            coef.columns = self.reg_vars
            insig_cols = coef.loc[:, (coef == 0).all()]
            insig_cols = insig_cols.columns
            
            model = model.drop(columns = insig_cols)
            pred = pred.drop(columns = insig_cols)
            
            self.model_data[key] = model
            self.pred_data[key] = pred
            
    def model_construction(self, model, param_grid):
        if self.use_training == True:
            self.model_params = {}
            self.model_estimator = {}
            self.model_performance = pd.DataFrame()
            for key, value in self.model_data.items():
                if len(value) <= 5: #branche must be deleted after full implementation
                    for i in range(2):
                        value = value.append(value, ignore_index = False)
                regs = value.drop(columns = 'Price')
                target = value['Price']
                X_train, X_test, y_train, y_test = train_test_split(regs, 
                                                                    target, 
                                                                    test_size=0.2, 
                                                                    random_state=11)
                
                grid_search = GridSearchCV(model, param_grid,
                                           scoring = 'neg_mean_squared_error',
                                           cv = 5)
                
                grid_search.fit(X_train, y_train)
                params = grid_search.best_params_
                estimator = grid_search.best_estimator_
                
                pred = estimator.predict(X_test)
                pred = pd.DataFrame(pred,
                                    columns = ['Prediction'],
                                    index = y_test.index)
                pred = pred.sort_index()
                y_test = pd.DataFrame(y_test)
                y_test = y_test.sort_index()
                mse = mean_squared_error(y_test, pred)
                mape = mean_absolute_percentage_error(y_test, pred)
                
                self.model_performance.loc[key, 'MSE'] = mse
                self.model_performance.loc[key, 'MAPE'] = mape
                self.model_params[key] = params
                self.model_estimator[key] = estimator
                
                plt.rc('figure', figsize = (20,10))
                plt.plot(pred['Prediction'], 
                         'b', lw = 1.5,
                         label = 'Prediction')
                plt.plot(y_test['Price'],
                         'r', lw = 1.5,
                         label = 'Actual Price')
                plt.legend(loc = 0, prop = legend_size)
                plt.grid(True)
                plt.xlabel('Date', fontdict = xlabel_size)
                plt.ylabel('Price', fontdict = ylabel_size)
                plt.title(f'{key} Test Set Evaluation', 
                          fontdict = title_size)
                plt.show()
                
            mean_mse = self.model_performance['MSE'].mean()
            mean_mape = self.model_performance['MAPE'].mean()
            self.model_performance.loc['Mean', 'MSE'] = mean_mse
            self.model_performance.loc['Mean', 'MAPE'] = mean_mape
            
            print('\n')
            print('Model Test Set Evaluation',
                  '\n')
            print(tabulate(self.model_performance,
                           headers = 'keys',
                           showindex = True,
                           tablefmt = 'simple',
                           numalign = 'center',
                           floatfmt = ('.2f')),
                  '\n\n')
                
            save_obj(self.model_estimator,
                     path,
                     file_best_est)
        else:
            self.model_estimator = GetFile(path, file_best_est)
            
    def target_prediction(self, raw_funds, quotes):
        self.pred_results = pd.DataFrame()
        cols = ['Reference Date', 'Report Date',
                'Price', 'Prediction', 'Return']
        for key, value in self.pred_data.items():
            estimator = self.model_estimator[key]
            target_pred = estimator.predict(value)
    
            raw_funds_adj = raw_funds[key].iloc[-1, :]
            ref_date = raw_funds_adj.name
            ref_date = ref_date.strftime('%Y-%m-%d')
            rep_date = raw_funds_adj.loc['fillingDate']
            rep_date = rep_date.strftime('%Y-%m-%d')
            price = quotes[key].iloc[-1, :]
            price = price['Adj Close']
            
            stock_results = pd.DataFrame(index = [key])
            stock_results.loc[:, 'Reference Date'] = ref_date
            stock_results.loc[:, 'Report Date'] = rep_date
            stock_results.loc[:, 'Price'] = price
            stock_results.loc[:, 'Prediction'] = target_pred
            stock_results.loc[:, 'Return'] = np.log(target_pred / price)
            
            self.pred_results = self.pred_results.append(stock_results)

        print('\n')
        print(tabulate(self.pred_results,
                       headers = 'keys',
                       showindex = True,
                       tablefmt = 'simple',
                       numalign = 'center',
                       floatfmt = ('.2f')),
              '\n\n')
            

            

        

    
    