# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 20:02:17 2020

@author: Administrator
"""

import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
import pickle
#%%
df = pd.read_csv('/Users/Administrator/.spyder-py3/DC_Properties.csv')
dfNull=  df.isnull().sum()
#%%
print('Percent of missing "Price" records is %.2f%%' %((df['PRICE'].isnull().sum()/df.shape[0])*100))
#%%
#Percent of missing "Price" records is 38.21%
total = df.isnull().sum().sort_values(ascending=False)
percent_1 = df.isnull().sum()/df.isnull().count()*100
percent_2 = (round(percent_1, 1)).sort_values(ascending=False)
missing_data = pd.concat([df.count(),total, percent_2], axis=1, keys=['Sum','Total', '%'])
df_NULLdataset=missing_data
#%%
dummy_dataset = df

dummy_dataset['Price_Flag'] = np.where(dummy_dataset.PRICE > 0 , 1,0)

unknown_dataset = dummy_dataset[dummy_dataset.Price_Flag != 1]

unknown_dataset.shape
dataset = dummy_dataset[dummy_dataset.Price_Flag != 0]
dataset.corr()
#%%
df=dataset
df.drop(['Unnamed: 0', "CMPLX_NUM", "LIVING_GBA" , "ASSESSMENT_SUBNBHD", "CENSUS_TRACT", 
         "CENSUS_BLOCK", "GIS_LAST_MOD_DTTM", "SALE_NUM","STORIES", "USECODE", "CITY", 
         "STATE", "NATIONALGRID",'X','Y','SALEDATE'],axis=1,inplace=True)
#%%
df.dropna(subset=['AYB'],inplace=True)
group_remodel= df.groupby(['EYB','AYB']).mean()['YR_RMDL']
#%%
def applyRemodel(x):
    if pd.notnull(x['YR_RMDL']):
        return x['YR_RMDL']
    else:
        return round(group_remodel.loc[x['EYB']][x['AYB']])
    
#%%
df['YR_RMDL'] = df[['YR_RMDL','EYB','AYB']].apply(applyRemodel,axis = 1)
df.dropna(subset=['YR_RMDL'],inplace=True)