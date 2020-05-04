# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 18:52:41 2020

@author: tejas
"""
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
df=Toyota_Corollacsv
toyota=df.iloc[:,[2,3,6,8,12,13,15,16,17]]
toyota.rename(columns={"Age_08_04":"Age"},inplace=True)
toyota.describe()

plt.boxplot(toyota["Price"]) #not normally distributed it is having right skew
plt.boxplot(toyota["Age"]) #not normally distributed it is having left skew     
plt.boxplot(toyota["HP"])  #not normally distributed it is having right skew
plt.boxplot(toyota["cc"])  # not normally distributed it s having right skew
plt.boxplot(toyota["Quarterly_Tax"]) # #not normally distributed it s having right skew
plt.boxplot(toyota["Weight"]) #not normally distributed it s having right skew
plt.boxplot(toyota["KM"])  #not normally distributed it is having right skew
plt.boxplot(toyota["Gears"]) #not normally distributed it is having slightly left skew
plt.boxplot(toyota["Doors"]) ##not normally distributed it is having  left skew

import statsmodels.api as sns
sns.graphics.qqplot(toyota["Price"],fit=True,line='45')#Data is not linear.
sns.graphics.qqplot(toyota["Age"],fit=True,line='45') #Data is not linear
sns.graphics.qqplot(toyota["KM"],fit=True,line='45') #Data is not linear
sns.graphics.qqplot(toyota["HP"],fit=True,line='45') #Data is not linear
sns.graphics.qqplot(toyota["cc"],fit=True,line='45')#Data is not linear
sns.graphics.qqplot(toyota["Quarterly_Tax"],fit=True,line='45')#Data is not linear
sns.graphics.qqplot(toyota["Weight"],fit=True,line='45')#Data is not linear
sns.graphics.qqplot(toyota["Gears"],fit=True,line='45')#Data is not linear

plt.hist(toyota["Price"])#Data is having right skew
plt.hist(toyota["Age"])#Data is having left skew
plt.hist(toyota["HP"])#Data is having right skew, uneven distribution.
plt.hist(toyota["cc"])#Data is having right skew
plt.hist(toyota["Quarterly_Tax"])#Data is having right skew
plt.hist(toyota["Weight"])#Data is having right skew

toyota.corr()
import seaborn as sn
sn.pairplot(toyota)
correlation_values= toyota.corr()

#Splitting the data into train & test 
from sklearn.model_selection import train_test_split
train_data,test_data=train_test_split(toyota)
train_data=train_data.reset_index()
test_data=test_data.reset_index()
train_data1=train_data.drop("index",axis=1)
test_data1=test_data.drop("index",axis=1)

import statsmodels.formula.api as smf
#Model=1
m1=smf.ols("Price~Age+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight", data= train_data1).fit()
m1.summary()
sns.graphics.influence_plot(m1)
#cc,Doors are insignificant as probability value is greater than 0.05 
#with influence index plot it is observed that observation 348,805 is an influencing index.
#removing  observation
train_data2=train_data1.drop(train_data1.index[[348,805]],axis=0)

#model=2
m2=smf.ols("Price~Age+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight",data= train_data2).fit()
m2.summary()
#Door is having probability value > 0.05 checking with influence index plot.
sns.graphics.influence_plot(m2)
#221 is an influencing index.

train_data3=train_data2.drop(train_data2.index[[348,805,468]],axis=0)

#model=3
m3=smf.ols("Price~Age+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight",data= train_data3).fit()
m3.summary() #r-squared=0.878
# As Doors is insignificant.

#Try checking with VIF values
rsq_doors=smf.ols("Doors~Age+KM+HP+Gears+Quarterly_Tax+Weight", data=train_data3).fit().rsquared
Vif_rsq=1/(1-rsq_doors) #1.18

#As VIF value of Doors is less than 10. But probability value is more than 0.05.

#Checking with AV Plots:
snf.graphics.plot_partregress_grid(m3)

#As preparing the model without Doors and checking the values:
#model4
m4=smf.ols("Price~Age+KM+HP+cc+Gears+Quarterly_Tax+Weight",data= train_data3).fit()
m4.summary() #r-squared=0.878

#As all the values are significant.

final_train_data=smf.ols("Price~Age+KM+HP+cc+Gears+Quarterly_Tax+Weight",data= train_data3).fit()
final_train_data.summary()

#Train model predaction
Train_pred = final_train_data.predict(train_data3)
#Train Residuals
Train_residuals=train_data3["Price"]-Train_pred
#Train Rmse
Train_rmse=np.sqrt(np.mean(Train_residuals*Train_residuals))

#Test Prediction
Test_pred=final_train_data.predict(test_data1)
#Test Residuals
Test_residuals=test_data1["Price"]-Test_pred
#Test Rmse
Test_rmse=np.sqrt(np.mean(Test_residuals*Test_residuals))

#Checking with original model
toyota1=toyota.drop(toyota.index[[80,221,601]], axis=0)

final_model=smf.ols("Price~Age+KM+HP+cc+Gears+Quarterly_Tax+Weight",data= toyota1).fit()
final_model.summary()  #rsquared-0.881
best_model=final_model.predict(toyota1)

#Linearity
plt.scatter(toyota1["Price"],best_model,c='r');plt.xlabel("Observed values");plt.ylabel("Predicted values")

# Residuals v/s Fitted values
plt.scatter(best_model,final_model.resid_pearson,c='r');plt.axhline(y=0,color='blue');plt.xlabel("Fitted values");plt.ylabel("Residuals")
## errors are kind off homoscadasticity i.e there is equal variance
###Normality
## histogram--- for checking if the errors are normally distributed or not.
plt.hist(final_model.resid_pearson) 
#Errors are normally distributed.


import pylab
import scipy.stats as st
st.probplot(final_model.resid_pearson, dist='norm',plot=pylab)
## Errors are normally distributed