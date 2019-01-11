# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import os
import numpy as np
from sklearn.neighbors.regression import  KNeighborsRegressor
from sklearn.linear_model import Lasso,LassoCV,LassoLarsCV
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import Binarizer  # 特征二值化 TODO
from sklearn.preprocessing import OneHotEncoder #独热编码 TODO

#pd.set_option('precision', 2)


# TODO
"""
暂时没有用到时间相关的特征
"""
not_select_features=['A5','A7','A9','A11','A14','A16','A20','A24','A25','A26','A28','B4','B5','B7','B9','B10','B11']

def onehot(train,test):
    all = pd.concat([train,test])
    enc = OneHotEncoder()
    enc.fit(all)
    return enc.transform(train),enc.transform(test)

def maxmin(x):
    from sklearn.preprocessing import MinMaxScaler
    minMax = MinMaxScaler()
    #将数据进行归一化
    minMax.fit(x)
    return minMax

def make_test_sample_mothed(data,rate = 0.3,n_samples = 5):
    resamples = []
    for i in range(n_samples):
        ires = data.sample(frac = rate,random_state = np.random.RandomState(2019))
        resamples.append(ires)
    return resamples

def make_test_specific_mothed(train_data,train_target,test_size=0.4, random_state=0):
    X_train,X_test, y_train, y_test = train_test_split(
            train_data,train_target,test_size=0.4, random_state=0)
    return X_train,X_test,y_train,y_test


def scaler(x,method = 'mm'):
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.preprocessing import Normalizer
    from sklearn.preprocessing import StandardScaler
    if method == 'mm':
        scaler = MinMaxScaler()
    elif method == 'norm':
        scaler = Normalizer() #在度量样本之间相似性时，如果使用的是二次型kernel，需要做Normalization
    elif method == 'std':
        scaler = StandardScaler()
    else:
        raise ValueError("not found %s" %method)
    #将数据进行归一化
    scaler.fit(x)
    return scaler


def scalerToOne(all_x,train_x,test_x,method = 'mm'):
    minMax = scaler(all_x)
    train_x = minMax.transform(train_x)
    test_x = minMax.transform(test_x)
    return train_x,test_x

def writeAns(test_data):
    test_data['rate']= test_data['rate'].apply(lambda x:round(x,3))
    import time
    date = time.strftime('%Y%m%d%H%M%S',time.localtime(time.time()))
    test_data[['id','rate']].to_csv(r'../ans/submit_'+date+'.csv',index = False,header = False,encoding = 'utf-8')
    print("Write down ans Ok!")