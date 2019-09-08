import quandl
import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, svm, cross_validation
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import datetime as dt


stock_name= raw_input("Hello! What is the stock that you want to look at?")

stock = "WIKI/" + stock_name

today=dt.date.today()
data = quandl.get(stock, start_date="2015-01-05", end_date=str(today), api_key="Q6yYWxwNmU7kqETE89Ds",column_index=4)

# 1) Show the data of stock of end price from 2015 forward
# data.plot();
# plt.ylabel('Price')
# plt.xlabel('Time')
# plt.show()


# 2) Set up for Prediction for 30 days into future
data['Prediction'] = data[['Close']].shift(-30)
X = np.array(data.drop(['Prediction'], 1))
X = preprocessing.scale(X)


X_forecast = X[-30:] # set X_forecast equal to last 30 days
X = X[:-30] # remove last 30 days  from X


y = np.array(data['Prediction'])
y = y[:-30]


# Perform linear regression
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = 0.5)

clf = LinearRegression()
clf.fit(X_train,y_train)

#print(data[0:])
forecast_prediction = clf.predict(X_forecast)
#print(data)
#print(forecast_prediction)
