# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.cross_validation import train_test_split

def imp():
    ds=pd.read_csv('coinbaseUSD_1-min_data_2014-12-01_to_2018-01-08.csv')
    del ds["Timestamp"]
    X=ds.iloc[:,:-1].values
    y=ds.iloc[:,-1].values
    
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=69)
    linreg=linear_model.LinearRegression()
    linreg.fit(X_train,y_train)
    print linreg.score(X_test,y_test)
    y_pred=linreg.predict(X_test)
    plt.plot(X_test,y_test,'g')
    plt.plot(X_test,y_pred,'r')
    plt.show()
imp()

