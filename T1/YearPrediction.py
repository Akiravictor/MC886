# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 10:47:20 2017

@author: VictorAkiraHassudaSi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

training_dataset = pd.read_csv('year-prediction-msd-train.csv')
X = training_dataset.iloc[:,1:13].values
Y = training_dataset.iloc[:,0].values

from sklearn.model_selection import train_test_split as tts
X_train, X_val, Y_train, Y_val = tts(X,Y,test_size = 0.2, random_state = 0)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)


from sklearn.linear_model import LinearRegression as lr
regressor = lr()
regressor.fit(X_scaled, Y_train)
Y_pred = regressor.predict(X_scaled)

from sklearn.metrics import mean_squared_error
error = mean_squared_error(Y_train, Y_pred)

learning_rate = 0.0001
iterations = 10

X_scaled = np.append(np.ones((len(X_scaled),1)), X_train,axis = 1)
coefs = np.append(regressor.intercept_, regressor.coef_)
J_hist = np.zeros((iterations,))

for i in range(iterations):
    m = Y_train.size
    #Y_pred = X_train.dot(coefs)
    
    for j in range(coefs.size):
        temp = X_scaled[:,j]
        temp.shape = (m,1)
        
        errors_x1 = np.zeros((Y_pred.size,1))
        diffs = Y_pred - Y_train
        #errors_x1 = diffs * temp
        for k in range(Y_pred.size):    
            errors_x1[k] = diffs[k] * temp[k]
        
        coefs[j] = coefs[j] - learning_rate * (1.0/m) * errors_x1.sum()
    
    Y_pred = X_scaled.dot(coefs)
    sqError = (Y_pred - Y_train)
    J_hist[i] = (1.0 / (2*Y_train.size)) * sqError.T.dot(sqError)
    
iters = np.array([1,2,3,4,5,6,7,8,9,10])
plt.plot(iters,J_hist,c='blue')

X_val = np.append(np.ones((len(X_val),1)), X_val,axis = 1)
Y_pred2 = X_val.dot(coefs)

final_error = np.sqrt(mean_squared_error(Y_val, Y_pred2))

#################################################

training_dataset = pd.read_csv('year-prediction-msd-test.csv')
X_test = training_dataset.iloc[:,1:13].values
Y_test = training_dataset.iloc[:,0].values

Y_pred_test = X_test.dot(coefs)
final_error_test = np.sqrt(mean_squared_error(Y_val, Y_pred2))