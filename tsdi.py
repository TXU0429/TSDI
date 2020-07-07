#!/usr/bin/env python
# coding: utf-8

# In[129]:


import numpy as np
import pandas as pd


# In[130]:


def get_data(df, category, time_range):
    """
    this function set up the data structure as a dataframe
    and enter existing data
    """
    
    #set up data structure
    new_df = pd.DataFrame()
    for i in df[category].unique():
        for n in range(np.min(df[time_range]), np.max(df[time_range])+1):
            new_df = new_df.append(pd.DataFrame(np.array([[i,n]]), columns = [category, time_range]))
    new_df = new_df.sort_values([category, time_range])
    
    #concat the old data with the new data structure
    new_df = new_df.set_index([category, time_range])
    df = df.set_index([category, time_range])
    result = pd.concat([new_df.reset_index(), df.reset_index(drop = True)], axis = 1, sort = False)
    return result


# In[131]:


df_grade = pd.read_csv('/Users/tongxu/Desktop/grades.csv', delimiter='\t')


# In[132]:


def ADF(df, category, time_range):
    """
    use the Augmented Dickey-Fuller(ADF) test to test the stationarity 
    null hypothesis: non-stationary
    alternative hypothesis: stationary
    """
    
    from statsmodels.tsa.stattools import adfuller
    
    df_stationarity = pd.DataFrame()
    for i in df[category].unique():
        for column in df.columns:
            if column != category and column != time_range:
                column_value = df[column].dropna().values
                column_result = adfuller(column_value)
                p_value = column_result[1]
                if p_value < 0.1:
                    df_stationarity = df_stationarity.append(pd.DataFrame(np.array([[i,column,'stationary']]), 
                                                                          columns = [category, 'Column Name', 'Stationarity']))
                else:
                    df_stationarity = df_stationarity.append(pd.DataFrame(np.array([[i,column,'not stationary']]), 
                                                                          columns = [category, 'Column Name', 'Stationarity']))
    return df_stationarity


# In[133]:


def Order_Permutation(df, category, time_range):
    """
    this function produce all possible permutation of ARIMA order (p,d,q)
    """
    
    from itertools import product
    
    df_order = pd.DataFrame()
    df_s = ADF(get_data(df, category, time_range), category, time_range)
    DoF = np.max(df[time_range])+1 - np.min(df[time_range])
    for index, row in df_s.iterrows():
        if row['Stationarity'] == 'stationary':
            maxRange = np.array([DoF, DoF, DoF])
            states = np.array([i for i in product(*(range(i+1) for i in maxRange)) if sum(i) <= DoF])
            for s in states:
                df_order = df_order.append(pd.DataFrame(np.array([[row[category],row['Column Name'],row['Stationarity'], s]]), 
                                                        columns = [category, 'Column Name', 'Stationarity','Order']))
        else:
            df_order = df_order.append(pd.DataFrame(np.array([[row[category], row['Column Name'], row['Stationarity'], 'no order']]), 
                                                    columns = [category, 'Column Name', 'Stationarity','Order']))
    return df_order


# In[141]:


SES(df_grade, 'Student', 'Year', "History")


# In[134]:


def SES(df, category, time_range, column):
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing
    new_df = pd.DataFrame()
    for u in df[category].unique():
        df2 = df[df[category] == u]
        df2 = df2.reset_index(drop=True)
        DoF = np.max(df[time_range])+1 - np.min(df[time_range])
        if df2.isna().sum()[column] == DoF:
            df2[column] = df[column].dropna().mean()
        while df2.isna().sum()[column] > 0: 
            data = df2[column].tolist()
            for n, nxt in zip (data, data[1:]):
                if np.isnan(n) == False and np.isnan(nxt) == True:
                    data_list = list()
                    for m in data[:data.index(nxt)]:
                        if np.isnan(m) == False:
                            data_list.append(m)
                    if len(data_list) < 2:
                        df2[column][data.index(nxt)] = n
                    else:
                        model = SimpleExpSmoothing(data_list)
                        model_fit = model.fit()
                        #obtain predicted value
                        yhat = model_fit.forecast()
                        df2[column][data.index(nxt)] = yhat
            data = df2[column].tolist()[::-1]
            for n, nxt in zip (data, data[1:] ):
                if np.isnan(n) == False and np.isnan(nxt) == True:
                    datalist = data[:data.index(nxt)]
                    data_list = list()
                    for m in datalist:
                        if np.isnan(m) == False:
                            data_list.append(m)
                    if len(data_list) < 2:
                        df2[column][len(data) - 1 - data.index(nxt)] = n
                    else:
                        model = SimpleExpSmoothing(data_list)
                        model_fit = model.fit()
                        #obtain predicted value
                        yhat = model_fit.forecast()
                        df2[column][len(data) - 1 - data.index(nxt)] = yhat
        new_df = new_df.append(df2)
    return new_df


# In[144]:


def Estimate(df, category, time_range, column):
    from statsmodels.tsa.arima_model import ARIMA
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing
    new_df = pd.DataFrame()
    df_MSE = pd.DataFrame()
    df_cm = ADF(df, category, time_range)
    df_op = Order_Permutation(df, category, time_range)
    for y in df[category].unique():
        df_cd = df_cm[(df_cm[category] == y) & (df_cm['Column Name'] == column)]
        stationarity_c = df_cd['Stationarity']
        stationarity_c = stationarity_c.iloc[0]
        if stationarity_c == "stationary": 
            print(y, column, 'stationary')
            df2 = df[df[category] == y]
            df2 = df2.reset_index()
            while df2.isna().sum()[column] > 0: 
                for index, row in df_op.iterrows():
                    if row[category] == y and row['Column Name'] == column:
                        df3 = df2
                        data = df3[column].tolist()
                        for n, nxt in zip (data, data[1:] ):
                            if np.isnan(n) == False and np.isnan(nxt) == True:
                                datalist = data[:data.index(nxt)]
                                data_list = list()
                                for m in datalist:
                                    if np.isnan(m) == False:
                                        data_list.append(m)
                                train_data = data_list[:-1]
                                test_data = data_list[-1:]
                                model = ARIMA(train_data, tuple(row['Order']))
                                model_fit = model.fit(disp = False)
                                yhat = model_fit.predict(len(train_data), len(train_data))
                                squared_error = np.square(test_data - yhat)
                                df_MSE = df_MSE.append(pd.DataFrame(np.array([[y, column, row['Order'], squared_error]]), columns = [category, 'Column Name', 'Order', 'MSE']))
                                df3[column][data.index(nxt)] = yhat
                data = df2[column].tolist()
                for n, nxt in zip (data, data[1:] ):
                    if np.isnan(n) == False and np.isnan(nxt) == True:
                        datalist = data[:data.index(nxt)]
                        data_list = list()
                        for m in datalist:
                            if np.isnan(m) == False:
                                data_list.append(m)
                        df_each = df_MSE[(df_MSE[category]==y) & (df_MSE['Column Name']==column)]
                        df_each_number = pd.to_numeric(df_each['MSE or Error'], errors='coerce')
                        order_length = len(df_each_number)
                        order_array = df_each['Order'][df_each['MSE or Error'] == df_each_number.min()]
                        order = tuple(order_array.iloc[0])
                        print('NaN location %s %s %s' %(y, column, nxt))
                        print('Best Model: ARIMA %s' %(order))
                        model = ARIMA(data_list, order)
                        model_fit = model.fit(disp=False, transparams=False)
                        yhat = model_fit.predict(len(data_list), len(data_list))
                        df2[column][data.index(nxt)] = yhat
                data = df2[column].tolist()[::-1]
                for n, nxt in zip (data, data[1:] ):
                    if np.isnan(n) == False and np.isnan(nxt) == True:
                        datalist = data[:data.index(nxt)]
                        data_list = list()
                        for m in datalist:
                            if np.isnan(m) == False:
                                data_list.append(m)
                        #analyze using ARIMA
                        df_each = df_MSE[(df_MSE[category]==y) & (df_MSE['Column Name']==column)]
                        df_each_number = pd.to_numeric(df_each['MSE or Error'], errors='coerce')
                        order_length = len(df_each_number)
                        order_array = df_each['Order'][df_each['MSE or Error'] == df_each_number.min()]
                        order = tuple(order_array.iloc[0])
                        print('NaN location %s %s %s' %(y, column, nxt))
                        print('Best Model: ARIMA %s' %(order))
                        model = ARIMA(data_list, order)
                        model_fit = model.fit(disp=False, transparams=False)
                        #obtain predicted value
                        yhat = model_fit.predict(len(data_list), len(data_list))
                        df2[column][len(data) - 1 - data.index(nxt)] = yhat
            new_df = new_df.append(df2)
        elif stationarity_c == "not stationary":
            print(y,column, 'not stationary')
            print('Best model: Exponential Smoothing')
            df2 = df[df[category] == y]
            df3 = SES(df2, category, time_range, column)
            new_df = new_df.append(df3)
    return new_df


# In[145]:


def Run(df, category, time_range):
    for col in df.columns:
        if col != category and col !=time_range:
            df = Estimate(df, category, time_range, col)
    return df


# In[146]:


Run(df_grade, 'Student', 'Year')


# In[ ]:




