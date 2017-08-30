# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 21:46:05 2017

@author: Akira
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset= pd.read_csv('year-prediction-msd-train.csv')
X = dataset.iloc[:,1:13].values
Y = dataset.iloc[:,0].values

#Spliting into training set and validation set
from sklearn.cross_validation import train_test_split as tts
X_train, X_test, Y_train, Y_test = tts(X,Y,test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression as lr
regressor = lr()
regressor.fit(X_train, Y_train)

Y_pred = regressor.predict(X_test)

from sklearn.metrics import mean_squared_error as mse

lin_mse = mse(Y_test,Y_pred)
lin_rmse = np.sqrt(lin_mse)


from sklearn.model_selection import cross_val_score

scores = cross_val_score(regressor, X_train, Y_train, scoring="neg_mean_squared_error", cv=10)

rmse_scores = np.sqrt(-scores)
rmse_scores.std()

plt.scatter(X_train, Y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.show()



#Building Backward Elimitation
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((463715,1)).astype(int), values = X, axis = 1)




#Scaling with Standardization
from sklearn.preprocessing import StandardScaler as ss
sc_X = ss()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)