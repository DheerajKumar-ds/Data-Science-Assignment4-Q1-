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

#Performing Transformation on each column

#Log Transformation
fig, ax=plt.subplots(2, figsize=(6,4), sharex= False, sharey = False)
sns.boxplot(np.log(df["Sorting Time"]), ax=ax[0])
sns.boxplot(np.log(df["Delivery Time"]), ax=ax[1])
plt.suptitle("Log Transformation on Continuous Variables", fontsize= 17, y = 1.06)
plt.tight_layout(pad=2.0)

#SquareRoot Transformation
fig, ax=plt.subplots(2, figsize=(6,4), sharex= False, sharey = False)
sns.boxplot(np.sqrt(df["Sorting Time"]), ax=ax[0])
sns.boxplot(np.sqrt(df["Delivery Time"]), ax=ax[1])
plt.suptitle("Sqrt Transformation on Continuous Variables", fontsize= 17, y = 1.06)
plt.tight_layout(pad=2.0)

#Cuberoot Transformation
fig, ax=plt.subplots(2, figsize=(6,4), sharex= False, sharey = False)
sns.boxplot(np.cbrt(df["Sorting Time"]), ax=ax[0])
sns.boxplot(np.cbrt(df["Delivery Time"]), ax=ax[1])
plt.suptitle("Cbrt Transformation on Continuous Variables", fontsize= 17, y = 1.06)
plt.tight_layout(pad=2.0)


#Correlation Analysis
df.corr()

#Split the variables into X and Y
x= df[['Sorting Time']]
y= df['Delivery Time']

#Model Fitting
import statsmodels.api as smf
import statsmodels.formula.api as sm
model = sm.ols('y~x', data = df).fit()
model.summary()

#Transformations
#Square Root transformation on data
model1 = sm.ols('np.sqrt(y)~np.sqrt(x)', data = df).fit()
model1.summary()

#Cube Root transformation on Data
model2 = sm.ols('np.cbrt(y)~np.cbrt(x)', data = df).fit()
model2.summary()

#Log transformation on Data
model3 = sm.ols('np.log(y)~np.log(x)', data = df).fit()
model3.summary()

#Parameters
model.params            #Coefficient/slope(m) and #Bias/Constant/Intercept(c)

#Metrics of model without transformations
Model_r2 = model.rsquared
Model_r2_adj = model.rsquared_adj


#Model Validation
#Comparing different models with respect to their Root Mean Squared Errors

from sklearn.metrics import mean_squared_error

model1_pred_y =np.square(model1.predict(x))
model2_pred_y =pow(model2.predict(x),3)
model3_pred_y =np.exp(model3.predict(x))

model1_rmse =np.sqrt(mean_squared_error(y, model1_pred_y))
model2_rmse =np.sqrt(mean_squared_error(y, model2_pred_y))
model3_rmse =np.sqrt(mean_squared_error(y, model3_pred_y))
print('model=', np.sqrt(model.mse_resid),'\n' 'model1=', model1_rmse,'\n' 'model2=', model2_rmse,'\n' 'model3=', model3_rmse)

data = {'model': np.sqrt(model.mse_resid), 'model1': model1_rmse, 'model2': model3_rmse, 'model3' : model3_rmse}
min(data, key=data.get)

#As model2 has the minimum RMSE and highest Adjusted R-squared score. Hence, we are going to use model2 to predict our values

#Predict the Values
Y_pred = model2.predict(x)
print(Y_pred)

#EDA after building model with pred values
import matplotlib.pyplot as plt
plt.scatter(x,y)
plt.scatter(x,model2_pred_y)                  #scatterplot
plt.plot(x,model2_pred_y,color='green')       #lineplot
sns.regplot(x,model2_pred_y)                  #regplot
plt.show()

#Metrics
from sklearn.metrics import mean_squared_error
MSE = mean_squared_error(y,model2_pred_y)
print("Mean Squared Error =", MSE)

RMSE = np.sqrt(MSE)
print("Root Mean Squared Error =", RMSE.round(2))

