# -*- coding: utf-8 -*-
"""
Created on Sat May  2 18:46:59 2020

@author: Debanjan

Creating Prediction time series for cumulative dataset
"""

import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
dataset = pd.read_csv(r'C:\Users\Debanjan\Desktop\novel-corona-virus-2019-dataset (3)\PredictedResult.csv',encoding = 'ISO-8859–1')
X = dataset.iloc[:,4].values.astype(float)

X = np.diff(X)
X = X.reshape(-1,1)
X = abs(X)
Xnew = np.cumsum(X)
Xnew= Xnew.reshape(-1,1)
Xnew = Xnew + 8889
plt.plot(Xnew)