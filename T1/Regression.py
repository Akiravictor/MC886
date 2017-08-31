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

#X_train = X_train.reshape((370972,12))
#X_test = X_test.reshape((92743,1))
#Y_train = Y_train.reshape((370972,1))
#Y_test = Y_test.reshape((92743,1))

#Multi Linear Regression
from sklearn.linear_model import LinearRegression as lr
regressor = lr()
regressor.fit(X_train, Y_train)

coefs = regressor.coef_ #thetas 1 - 11
intercept = regressor.intercept_ #theta 0

Y_linear_pred = regressor.predict(X_test)

from sklearn.metrics import mean_squared_error as mse

lin_mse = mse(Y_test,Y_linear_pred)
lin_rmse = np.sqrt(lin_mse)

#lin_rmse is too high -> 10.07

alpha = 0.1
n_iterations = 100

for i in range(n_iterations):
    grad = 1/12 * X_train.T.dot(X.dot(coefs) - Y_train)
    coefs = coefs - alpha*grad











#Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(X_train, Y_train)

Y_tree_pred = tree_reg.predict(X_test)

tree_mse = mse(Y_test,Y_tree_pred)
tree_rmse = np.sqrt(tree_mse)

#Cross Validation
from sklearn.model_selection import cross_val_score

scores = cross_val_score(tree_reg, X_train, Y_train, scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)
rmse_scores.std()

#Building Backward Elimitation
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((463715,1)).astype(int), values = X, axis = 1)
X_opt = X[:, [0,1,2,3,4,5,6,7,8,9,10,11,12]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()


##### Manual Regression #####

