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
import feature_engineer as fe
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

#pd.set_option('precision', 2)


not_select_features=['A5','A7','A9','A11','A14','A16','A20','A24','A26','A28','B4','B5','B7','B9','B10','B11']

train_data = pd.read_csv(r'..\jinnan_round1_train_20181227.csv',encoding = 'gbk').fillna(0)
train_data.ix[1304,'A25'] = '0' 
train_data['A25'] = train_data['A25'] .astype('int')
train_data.ix[1304,'A25'] = train_data['A25'].mean()
test_data = pd.read_csv(r'..\jinnan_round1_testA_20181227.csv',encoding = 'gbk').fillna(0)
A25 = train_data['A25']
not_select = train_data[not_select_features]

select_features = np.setdiff1d(train_data.columns.tolist(),not_select_features).tolist()
y = 'rate'
select_features.remove(y)
select_features.remove('id')
train_y = train_data[y]
train_x = train_data[select_features]

test_x = test_data[select_features]

all_x = pd.concat([train_x,test_x],ignore_index=True)

index_train = np.arange(train_x.shape[0])
index_test = index_train+test_x.shape[0]

#-----------------we should select a base line--------------------------#
#70%作为训练 30%作为测试 随机种子为0
X_train,X_test,y_train,y_test = fe.make_test_specific_mothed(train_x,
                                                          train_y,
                                                          test_size=0.3,
                                                          random_state=0)

# ------------------------------------scaler-------------------------------------
#X_train,X_test = fe.scalerToOne(np.concatenate([X_train,X_test]),X_train,X_test,method = 'norm')
#train_x,test_x = fe.scalerToOne(all_x,train_x,test_x,method = 'norm')
#--------------------------SVR(参数不收敛)----------------------------------------------
# 自动选择合适的参数
#svr = GridSearchCV(SVR(kernel = "linear"), param_grid={"C": (0.1,1,10), "gamma": (0.1,1,10)},scoring = 'neg_mean_squared_error', n_jobs = 4,cv = 3,verbose = 1)
#svr.fit(train_x, train_y)
#print(svr.best_params_)
#--------------------------------------------------------------------------------
#---------------------------------random forest----------------------------------
clf_rf = RandomForestRegressor(n_estimators=250, criterion='mse', max_depth=5, min_samples_split=2, min_samples_leaf=3, min_weight_fraction_leaf=0.0,random_state = 92092617)

clf_rf.fit(X_train,y_train)

ans = clf_rf.predict(X_test)
print(mean_squared_error(y_test,ans))

clf_rf.fit(train_x,train_y)
rf=clf_rf.predict(test_x)
#---------------------------------gbdt-------------------------------------------
clf_gbdt = GradientBoostingRegressor(loss='huber', learning_rate=0.13, n_estimators=75, subsample=0.8, criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, max_depth=5,random_state=92092617, alpha=0.9)
#'ls', 'lad', 'huber', 'quantile'
clf_gbdt.fit(X_train,y_train)

ans = clf_gbdt.predict(X_test)
print(mean_squared_error(y_test,ans))

clf_gbdt.fit(train_x,train_y)
gbdt=clf_gbdt.predict(test_x)
test_data['rate'] = gbdt
fe.writeAns(test_data)
#------------------------------lassocv-------------------------------------------
#clf = LassoCV(cv= 2)
#clf.fit(train_x, train_y)
#--------------------------------------------------------------------------------





#-------------------------------------knn----------------------------------------
#clf = KNeighborsRegressor(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=1)
##--------------------------------------------------------------------------------
#clf.fit(X_train,y_train)
#ans = clf.predict(X_test)
#print(mean_squared_error(y_test,ans))
#
#clf.fit(train_x,train_y)
#knn =clf.predict(test_x)
#test_data['rate'] = (rf + knn)/2
##---------------------------cross validation-------------------------------------
#clf_gbdt = GradientBoostingRegressor(loss='huber', learning_rate=0.13, n_estimators=75, subsample=0.8, criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, max_depth=5,random_state=92092617, alpha=0.9)
#cv = cross_val_score(clf_gbdt, train_x, train_y, scoring='neg_mean_squared_error', cv=5, verbose=1)
#
#print(np.mean(cv))
#
#fe.writeAns(test_data)