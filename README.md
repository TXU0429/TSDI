# TSDI
Time Series Data Imputation using ARIMA and Exponential Smoothing

# Library
* import numpy as np
* import pandas as pd
* from statsmodels.tsa.stattools import adfuller
* from itertools import product
* from statsmodels.tsa.holtwinters import SimpleExpSmoothing
* from statsmodels.tsa.arima_model import ARIMA

# Input
dataframe with one category column, one time_range column, other columns being the attributes of numberic values (and some missing values). 

# Output
dataframe without missing values. 

# Procedures
* step 1: Search for categories' names and the longest time range (e.g., from year 2000 to year 2020) and set up a dataframe being the complete structure with some missing values that need to be fulfilled. 
* step 2: Fill the dataframe with existing data points. 
* step 3: Cut the data into slides with each slide of one category throughout the time range, and one attribute with missing values. 
* step 4: Use ADF test to see if the slide of data is stationary. 
* step 5: List all possible permutations of ARIMA model parameters (p,d,q) and test each set of parameters on each slide of data with the last one of non-NaN data being the test data point and the rest non-NaN data points being the training data. 
* step 6: For each slide, select the set of parameters with smallest mean square error. 
* step 7: Use that set of parameters to estimate missing values. If not stationary or without enough degree of freedom, use Exponential Smoothing to estimate missing values. 
* step 8: Return a dataframe without missing values. 

# Limitations
* Right now, only supprt time range of year. Month, day, hour, minute, second...other time ranges to be added. 
