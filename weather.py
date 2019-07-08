#!/usr/bin/env python
# coding: utf-8

# In[34]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df=pd.read_excel('C:/Users/user/Downloads/Data.xls')
df.info()
df.describe()
df.corr()
col_drop=["REGION"]
df=df.drop(columns=col_drop,axis=1)
fig,ax=plt.subplots(1,1)
ax.scatter(df.YEAR,df.JAN)
ax.set_xlabel('YEAR')
ax.set_ylabel('JAN')
plt.show()
fig,ax=plt.subplots(1,1)
ax.scatter(df.YEAR,df.FEB)
ax.set_xlabel('YEAR')
ax.set_ylabel('FEB')
plt.show()
fig,ax=plt.subplots(1,1)
ax.scatter(df.YEAR,df.MAR)
ax.set_xlabel('YEAR')
ax.set_ylabel('MAR')
plt.show()
fig,ax=plt.subplots(1,1)
ax.scatter(df.YEAR,df.APR)
ax.set_xlabel('YEAR')
ax.set_ylabel('APRIL')
plt.show()
fig,ax=plt.subplots(1,1)
ax.scatter(df.YEAR,df.MAY)
ax.set_xlabel('YEAR')
ax.set_ylabel('MAY')
plt.show()
fig,ax=plt.subplots(1,1)
ax.scatter(df.YEAR,df.JUN)
ax.set_xlabel('YEAR')
ax.set_ylabel('JUN')
plt.show()
fig,ax=plt.subplots(1,1)
ax.scatter(df.YEAR,df.JUL)
ax.set_xlabel('YEAR')
ax.set_ylabel('JUL')
plt.show()
fig,ax=plt.subplots(1,1)
ax.scatter(df.YEAR,df.AUG)
ax.set_xlabel('YEAR')
ax.set_ylabel('AUG')
plt.show()
fig,ax=plt.subplots(1,1)
ax.scatter(df.YEAR,df.SEP)
ax.set_xlabel('YEAR')
ax.set_ylabel('SEP')
plt.show()
fig,ax=plt.subplots(1,1)
ax.scatter(df.YEAR,df.OCT)
ax.set_xlabel('YEAR')
ax.set_ylabel('OCT')
plt.show()
fig,ax=plt.subplots(1,1)
ax.scatter(df.YEAR,df.NOV)
ax.set_xlabel('YEAR')
ax.set_ylabel('NOV')
plt.show()
fig,ax=plt.subplots(1,1)
ax.scatter(df.YEAR,df.DEC)
ax.set_xlabel('YEAR')
ax.set_ylabel('DEC')
plt.show()

from sklearn import linear_model,feature_selection,preprocessing
from sklearn.model_selection import train_test_split
import statsmodels.formula.api as sm
from statsmodels.tools.eval_measures import mse
from statsmodels.tools import add_constant
from sklearn.metrics import mean_squared_error

X=df.values.copy()
X_train,X_valid,y_train,y_valid=train_test_split( X[:, :-1], X[:, -1], train_size=0.80)
result=sm.OLS(y_train,add_constant(X_train)).fit()
result.summary()
result=sm.OLS(y_train,add_constant(X_train)).fit()
result.summary()
ypred = result.predict(add_constant(X_valid))
print (mse(ypred,y_valid))
fig,ax=plt.subplots(1,1)
ax.scatter(y_valid,ypred)
ax.set_xlabel('Actual')
ax.set_ylabel('Prediction')
plt.show()


# In[ ]:





# In[ ]:




