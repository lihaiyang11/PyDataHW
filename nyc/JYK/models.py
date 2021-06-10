# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 00:46:06 2018

@author: dell
"""
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
#import warnings
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
#sklearn里面用来处理缺失值的包，使用均值、中位值或者缺失值所在列中频繁出现的值来替换
from sklearn.preprocessing import Imputer
from sklearn.linear_model import SGDRegressor
from sklearn import ensemble

#训练集为之前生成的result文件，包括了所有需要的特征值
train = pd.read_csv('result.csv', index_col = 0)
#print(train.head())
feature_means = train.columns.values.tolist()

do_not_use_for_training = ['pick_date', 'id', 'pickup_datetime', 'dropoff_datetime', 'trip_duration', 'store_and_fwd_flag']
feature_names = [f for f in train.columns if f not in do_not_use_for_training]
#print("We will use following features for training {}. " .format(feature_names))
#print("Total number of features are {}." .format(len(feature_names)))
#print("Number of Nulls train - {}.".format(train.isnull().sum().sum()))

X = train[feature_names]
Y = train['trip_duration']

#以占总体0.1的比例来划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X.values, Y, test_size = 0.1, random_state = 1987)

#检查训练集与测试集中包含NaN的数据项并删除
#imp = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0, verbose = 0, copy = True)
X_train = Imputer().fit_transform(X_train)
X_test = Imputer().fit_transform(X_test)

print("training")
start = time.time()
#调参见Tuning.py，最后还有一定的手动调参过程。
#主要的调整参数为n_estimators,max_depth,min_samples_leaf,min_samples_split,subsample
clf = ensemble.GradientBoostingRegressor(learning_rate=0.1, n_estimators=80, max_depth=7, 
                                         min_samples_leaf=80, min_samples_split=1000, 
                                         max_features='sqrt', subsample = 0.8,  random_state=10)
#放入训练集，建立模型
gbdt_model = clf.fit(X_train, y_train)

#对测试集进行预测
Y_predict = gbdt_model.predict(X_test)
print(Y_predict)
end = time.time()
print("Time taken is {}.".format(end - start))

#将ndarray格式转化成array
pd_Y_predict = np.array(Y_predict)
y_test = np.array(y_test)

#统计一下关于预测中是否存在不合理结果（预测值小于0）
negative_num = 0
for i in range(len(Y_predict)):
    if(Y_predict[i] <= 0):
        negative_num += 1
print("There are {} negative numbers in prediction".format(negative_num))

"""
其余尝试：
#SGD Regressor 随机梯度下降回归模型，预测效果不佳，有很多负数
sgdr = SGDRegressor()
sgdr.fit(X_train, y_train)
Y_predict = sgdr.predict(X_test)
print("The value of default measurement of SGDRegressor is {}.".format(sgdr.score(X_test, y_test)))

#adaboost calssifier 分类器不适用于大量数据的回归分析，内存炸
print("training")
model1 = DecisionTreeClassifier(max_depth = 1)
model = AdaBoostClassifier(model1, n_estimators = 50)
model.fit(X_train, y_train)
print("predict")
Y_predict = np.array(model.predict(X_test))
"""

#计算所得的RMSLE值，如果有负数，替换为0
sum_pa = 0
for i in range(len(Y_predict)):
    if(Y_predict[i] <= 0):
        sum_pa += (np.log(y_test[i] + 1))**2
    else:
        sum_pa += (np.log(Y_predict[i] + 1) - np.log(y_test[i] + 1))**2
result = np.sqrt(sum_pa / len(Y_predict))
print("RMSLE:{}".format(result))

"""
样例输出：
training
[ 3619.15629558   442.24375105   518.2408557  ...,   678.12061556
  1389.55199765  1944.84735595]
Time taken is 1178.74906873703.
There are 18 negative numbers in prediction
RMSLE:0.5160191960636279
"""
