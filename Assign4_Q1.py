# -*- coding: utf-8 -*-
"""

@author: user
"""

import numpy as np
import pandas as pd

#Importing Dataset
df = pd.read_csv("delivery_time.csv")

#Exploratory Data Analysis
import matplotlib.pyplot as plt
import seaborn as sns

#scatterplot
plt.scatter(df['Delivery Time'], df['Sorting Time'])
plt.xlabel("Delivery Time")
plt.ylabel("Sorting Time")
plt.show()

#distplot
sns.distplot(df['Delivery Time'])
sns.distplot(df['Sorting Time'])

#Correlation Analysis
df.corr()

#Split the variables into X and Y
x= df[['Sorting Time']]
y= df['Delivery Time']

#Model Fitting
from sklearn.linear_model import LinearRegression
LR= LinearRegression()
LR.fit(x,y)
sns.regplot(x,y)

#Parameters
LR.coef_            #Coefficient/slope(m)
LR.intercept_       #Bias/Constant(c)

#Predict the Values
Y_pred = LR.predict(x)
print(Y_pred)

#EDA after building model with pred values
import matplotlib.pyplot as plt
plt.scatter(x,y)
plt.scatter(x,Y_pred)                  #scatterplot
plt.plot(x,Y_pred,color='green')       #lineplot
sns.regplot(x,Y_pred)                  #regplot
plt.show()

#Metrics
from sklearn.metrics import mean_squared_error
MSE = mean_squared_error(y,Y_pred)
print("Mean Squared Error =", MSE)

RMSE = np.sqrt(MSE)
print("Root Mean Squared Error =", RMSE.round(2))


"MODEL PREDICTIONS"
# Manual prediction for say sorting time 5
delivery_time = (6.582734) + (1.649020)*(5)
delivery_time

# Automatic Prediction for say sorting time 5, 8
new_data=pd.Series([5,8])
new_data

data_pred=pd.DataFrame(new_data,columns=['sorting_time'])
data_pred

#Predicted Delivery Time
LR.predict(data_pred)