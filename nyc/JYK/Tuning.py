# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 13:32:58 2018

@author: dell
"""

import time
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
#sklearn里面用来处理缺失值的包，使用均值、中位值或者缺失值所在列中频繁出现的值来替换
from sklearn.preprocessing import Imputer
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.grid_search import GridSearchCV

def RMSLE(y_predict, y_test):
    
    negative_num = 0
    for i in range(len(y_predict)):
        if(y_predict[i] <= 0):
            negative_num += 1
    print("There are {} negative numbers in prediction".format(negative_num))
    
    sum_pa = 0
    for i in range(len(y_predict)):
        if(y_predict[i] <= 0):
            sum_pa += (np.log(y_test[i] + 1))**2
        else:
            sum_pa += (np.log(y_predict[i] + 1) - np.log(y_test[i] + 1))**2
    result = np.sqrt(sum_pa / len(y_predict))
    return result

train = pd.read_csv('result.csv', index_col = 0)
del train['Unnamed: 0.1']
del train['Unnamed: 0.1.1']
train = train.iloc[0:100000]
feature_means = train.columns.values.tolist()

do_not_use_for_training = ['pick_date', 'id', 'pickup_datetime', 'dropoff_datetime', 'trip_duration', 'store_and_fwd_flag']
feature_names = [f for f in train.columns if f not in do_not_use_for_training]
print("We will use following features for training {}. " .format(feature_names))
print("Total number of features are {}." .format(len(feature_names)))

X = train[feature_names]
y = train['trip_duration']
X = Imputer().fit_transform(X)

#由于预测中会出现零星负值，所以不方便使用neg_mean_squared_log_error来做scoring进行调参，我们使用neg_mean_squared_error作为scoring方式


#n_estimators
gsearch1 = GridSearchCV(estimator = GradientBoostingRegressor(learning_rate = 0.1, min_samples_split = 300, min_samples_leaf = 20, max_depth = 8, max_features = 'sqrt', subsample = 0.8, random_state = 10),
                        param_grid = {'n_estimators': [20,30,40,50,60,70,80]}, 
                        scoring = 'neg_mean_squared_error', iid = False, cv = 5)
gsearch1.fit(X, y)
print(gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_)

"""
[mean: -8815162.65277, std: 1194558.85623, params: {'n_estimators': 20}, 
 mean: -8816442.05210, std: 1190649.11365, params: {'n_estimators': 30}, 
 mean: -8816558.35348, std: 1189033.71494, params: {'n_estimators': 40}, 
 mean: -8819325.65246, std: 1188347.56428, params: {'n_estimators': 50}, 
 mean: -8821981.65202, std: 1188295.48681, params: {'n_estimators': 60}, 
 mean: -8825424.22265, std: 1185990.80746, params: {'n_estimators': 70}, 
 mean: -8828996.94165, std: 1186707.16514, params: {'n_estimators': 80}] 
 {'n_estimators': 20} -8815162.652774245
"""

#对决策树进行调参。首先我们对决策树最大深度max_depth和内部节点再划分所需最小样本数min_samples_split进行网格搜索。
param_test2 = {'max_depth':range(3,14,2), 'min_samples_split':range(100,801,200)}
gsearch2 = GridSearchCV(estimator = GradientBoostingRegressor(learning_rate=0.1, n_estimators=20, min_samples_leaf=20, max_features='sqrt', subsample=0.8, random_state=10), 
                        param_grid = {'max_depth':[3,5,7,9,11,13], 'min_samples_split':[100,200,300,400,500,600,700,800]}, scoring='neg_mean_squared_error',iid=False, cv=5)
gsearch2.fit(X,y)
print(gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_)

"""
[mean: -8819770.47204, std: 1195555.80350, params: {'max_depth': 3, 'min_samples_split': 100}, 
 mean: -8819569.60139, std: 1195757.47337, params: {'max_depth': 3, 'min_samples_split': 200}, 
 mean: -8819126.21014, std: 1195633.29525, params: {'max_depth': 3, 'min_samples_split': 300}, 
 mean: -8818691.22595, std: 1195042.77755, params: {'max_depth': 3, 'min_samples_split': 400}, 
 mean: -8819640.68908, std: 1195493.05815, params: {'max_depth': 3, 'min_samples_split': 500}, 
 mean: -8819449.33719, std: 1195565.34433, params: {'max_depth': 3, 'min_samples_split': 600}, 
 mean: -8819285.09427, std: 1195456.71234, params: {'max_depth': 3, 'min_samples_split': 700}, 
 mean: -8819421.25971, std: 1196075.27402, params: {'max_depth': 3, 'min_samples_split': 800}, 
 mean: -8811019.71973, std: 1201583.66110, params: {'max_depth': 5, 'min_samples_split': 100}, 
 mean: -8813843.04446, std: 1197921.38230, params: {'max_depth': 5, 'min_samples_split': 200}, 
 mean: -8816466.39018, std: 1198212.69395, params: {'max_depth': 5, 'min_samples_split': 300}, 
 mean: -8813334.75644, std: 1197156.44335, params: {'max_depth': 5, 'min_samples_split': 400}, 
 mean: -8813924.03427, std: 1196142.96911, params: {'max_depth': 5, 'min_samples_split': 500}, 
 mean: -8812329.70973, std: 1197556.86763, params: {'max_depth': 5, 'min_samples_split': 600}, 
 mean: -8812759.79380, std: 1196805.90767, params: {'max_depth': 5, 'min_samples_split': 700}, 
 mean: -8812491.77158, std: 1196775.68973, params: {'max_depth': 5, 'min_samples_split': 800}, 
 mean: -8821129.79260, std: 1196766.85033, params: {'max_depth': 7, 'min_samples_split': 100}, 
 mean: -8813534.33330, std: 1195699.44015, params: {'max_depth': 7, 'min_samples_split': 200}, 
 mean: -8806178.78829, std: 1195626.19919, params: {'max_depth': 7, 'min_samples_split': 300}, 
 mean: -8806215.25824, std: 1193475.86484, params: {'max_depth': 7, 'min_samples_split': 400}, 
 mean: -8806707.06895, std: 1197789.50247, params: {'max_depth': 7, 'min_samples_split': 500}, 
 mean: -8806961.54352, std: 1194067.58522, params: {'max_depth': 7, 'min_samples_split': 600}, 
 mean: -8806561.51544, std: 1195715.00967, params: {'max_depth': 7, 'min_samples_split': 700}, 
 mean: -8807512.38374, std: 1196180.75728, params: {'max_depth': 7, 'min_samples_split': 800}, 
 mean: -8822280.45950, std: 1198549.90764, params: {'max_depth': 9, 'min_samples_split': 100}, 
 mean: -8816569.89955, std: 1198910.28555, params: {'max_depth': 9, 'min_samples_split': 200}, 
 mean: -8820196.91763, std: 1195237.02834, params: {'max_depth': 9, 'min_samples_split': 300}, 
 mean: -8814048.69391, std: 1198224.99763, params: {'max_depth': 9, 'min_samples_split': 400}, 
 mean: -8815396.67807, std: 1196597.45308, params: {'max_depth': 9, 'min_samples_split': 500}, 
 mean: -8811891.65974, std: 1194003.89936, params: {'max_depth': 9, 'min_samples_split': 600}, 
 mean: -8814023.98652, std: 1202417.87477, params: {'max_depth': 9, 'min_samples_split': 700}, 
 mean: -8816585.97340, std: 1197555.07204, params: {'max_depth': 9, 'min_samples_split': 800}, 
 mean: -8842542.29380, std: 1201850.24598, params: {'max_depth': 11, 'min_samples_split': 100}, 
 mean: -8818337.57650, std: 1188705.11642, params: {'max_depth': 11, 'min_samples_split': 200}, 
 mean: -8818846.74338, std: 1191898.51598, params: {'max_depth': 11, 'min_samples_split': 300}, 
 mean: -8811388.76499, std: 1195901.18231, params: {'max_depth': 11, 'min_samples_split': 400}, 
 mean: -8818665.84684, std: 1199834.86369, params: {'max_depth': 11, 'min_samples_split': 500}, 
 mean: -8817449.14749, std: 1191534.25658, params: {'max_depth': 11, 'min_samples_split': 600}, 
 mean: -8817698.65624, std: 1194281.69718, params: {'max_depth': 11, 'min_samples_split': 700}, 
 mean: -8815171.87140, std: 1193588.32312, params: {'max_depth': 11, 'min_samples_split': 800}, 
 mean: -8847459.60511, std: 1191775.62202, params: {'max_depth': 13, 'min_samples_split': 100}, 
 mean: -8828332.93050, std: 1189548.79883, params: {'max_depth': 13, 'min_samples_split': 200}, 
 mean: -8818778.83693, std: 1199577.33770, params: {'max_depth': 13, 'min_samples_split': 300}, 
 mean: -8828074.68611, std: 1198479.71912, params: {'max_depth': 13, 'min_samples_split': 400}, 
 mean: -8818484.52678, std: 1199327.20856, params: {'max_depth': 13, 'min_samples_split': 500}, 
 mean: -8817698.15504, std: 1192195.58030, params: {'max_depth': 13, 'min_samples_split': 600}, 
 mean: -8818945.34308, std: 1194704.80889, params: {'max_depth': 13, 'min_samples_split': 700}, 
 mean: -8819486.11882, std: 1192995.67334, params: {'max_depth': 13, 'min_samples_split': 800}] 
{'max_depth': 7, 'min_samples_split': 300} -8806178.788289417
"""

#由于决策树深度7是一个比较合理的值，我们把它定下来，对于内部节点再划分所需最小样本数min_samples_split，我们暂时不能一起定下来，因为这个还和决策树其他的参数存在关联。
#下面我们再对内部节点再划分所需最小样本数min_samples_split和叶子节点最少样本数min_samples_leaf一起调参。
param_test3 = {'min_samples_split':range(800,1900,200), 'min_samples_leaf':range(60,101,10)}
gsearch3 = GridSearchCV(estimator = GradientBoostingRegressor(learning_rate=0.1, n_estimators=20, max_depth=7,max_features='sqrt', subsample=0.8, random_state=10), 
                       param_grid = {'min_samples_split':[800,1000,1200,1400,1600,1800], 'min_samples_leaf':[60,70,80,90,100]}, scoring='neg_mean_squared_error',iid=False, cv=5)
gsearch3.fit(X,y)
print(gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_)

"""
[mean: -8802762.91033, std: 1197760.90779, params: {'min_samples_leaf': 60, 'min_samples_split': 800}, 
 mean: -8797070.93923, std: 1199105.46653, params: {'min_samples_leaf': 60, 'min_samples_split': 1000}, 
 mean: -8800944.77267, std: 1197664.32286, params: {'min_samples_leaf': 60, 'min_samples_split': 1200}, 
 mean: -8798722.22007, std: 1199454.13055, params: {'min_samples_leaf': 60, 'min_samples_split': 1400}, 
 mean: -8801497.23099, std: 1198775.02571, params: {'min_samples_leaf': 60, 'min_samples_split': 1600}, 
 mean: -8801625.20683, std: 1195445.49590, params: {'min_samples_leaf': 60, 'min_samples_split': 1800}, 
 mean: -8798843.65920, std: 1198103.20581, params: {'min_samples_leaf': 70, 'min_samples_split': 800}, 
 mean: -8799116.81065, std: 1193595.51785, params: {'min_samples_leaf': 70, 'min_samples_split': 1000}, 
 mean: -8798263.76176, std: 1195170.18362, params: {'min_samples_leaf': 70, 'min_samples_split': 1200}, 
 mean: -8800877.47056, std: 1198755.26336, params: {'min_samples_leaf': 70, 'min_samples_split': 1400}, 
 mean: -8801016.18066, std: 1196408.74565, params: {'min_samples_leaf': 70, 'min_samples_split': 1600}, 
 mean: -8801299.99225, std: 1194538.45069, params: {'min_samples_leaf': 70, 'min_samples_split': 1800}, 
 mean: -8799406.54737, std: 1200027.96346, params: {'min_samples_leaf': 80, 'min_samples_split': 800}, 
 mean: -8798551.79866, std: 1196344.26901, params: {'min_samples_leaf': 80, 'min_samples_split': 1000}, 
 mean: -8796368.32092, std: 1195179.00400, params: {'min_samples_leaf': 80, 'min_samples_split': 1200}, 
 mean: -8797475.21020, std: 1198146.11434, params: {'min_samples_leaf': 80, 'min_samples_split': 1400}, 
 mean: -8800362.60487, std: 1198238.49452, params: {'min_samples_leaf': 80, 'min_samples_split': 1600}, 
 mean: -8801137.19701, std: 1195352.90680, params: {'min_samples_leaf': 80, 'min_samples_split': 1800}, 
 mean: -8798831.98623, std: 1201020.16106, params: {'min_samples_leaf': 90, 'min_samples_split': 800}, 
 mean: -8796444.05260, std: 1199489.96777, params: {'min_samples_leaf': 90, 'min_samples_split': 1000}, 
 mean: -8795771.21968, std: 1197933.68709, params: {'min_samples_leaf': 90, 'min_samples_split': 1200}, 
 mean: -8799632.30183, std: 1198783.78409, params: {'min_samples_leaf': 90, 'min_samples_split': 1400}, 
 mean: -8797750.82912, std: 1200106.98877, params: {'min_samples_leaf': 90, 'min_samples_split': 1600}, 
 mean: -8797573.06296, std: 1196504.40627, params: {'min_samples_leaf': 90, 'min_samples_split': 1800}, 
 mean: -8796586.86164, std: 1200318.38254, params: {'min_samples_leaf': 100, 'min_samples_split': 800}, 
 mean: -8799558.98436, std: 1196398.66814, params: {'min_samples_leaf': 100, 'min_samples_split': 1000}, 
 mean: -8798002.10593, std: 1195979.49459, params: {'min_samples_leaf': 100, 'min_samples_split': 1200}, 
 mean: -8800691.83787, std: 1201887.39660, params: {'min_samples_leaf': 100, 'min_samples_split': 1400}, 
 mean: -8796034.27513, std: 1197976.76175, params: {'min_samples_leaf': 100, 'min_samples_split': 1600}, 
 mean: -8798703.24725, std: 1196803.78922, params: {'min_samples_leaf': 100, 'min_samples_split': 1800}] 
{'min_samples_leaf': 90, 'min_samples_split': 1200} -8795771.21968346
"""

#放在模型里去测试啦
#GradientBoostingRegressor(learning_rate=0.1, n_estimators=20, max_depth=7, min_samples_leaf=90, min_samples_split=1200, max_features='sqrt', random_state=10)
#对最大特征数max_features进行网格搜索
param_test5 = {'subsample':[0.6,0.7,0.75,0.8,0.85,0.9]}
gsearch5 = GridSearchCV(estimator = GradientBoostingRegressor(learning_rate=0.1, n_estimators=20, max_depth=7, min_samples_leaf=90, min_samples_split=1200, max_features='sqrt', random_state=10), 
                       param_grid = {'subsample':[0.6,0.7,0.75,0.8,0.85,0.9]}, scoring='neg_mean_squared_error',iid=False, cv=5)
gsearch5.fit(X,y)
print(gsearch5.grid_scores_, gsearch5.best_params_, gsearch5.best_score_)
"""
[mean: -8802685.55311, std: 1197687.79264, params: {'subsample': 0.6}, 
 mean: -8799828.14929, std: 1199452.06668, params: {'subsample': 0.7}, 
 mean: -8798874.33381, std: 1201118.00052, params: {'subsample': 0.75}, 
 mean: -8795771.21968, std: 1197933.68709, params: {'subsample': 0.8}, 
 mean: -8797512.60784, std: 1194634.85075, params: {'subsample': 0.85}, 
 mean: -8796595.52189, std: 1195043.64082, params: {'subsample': 0.9}] 
{'subsample': 0.8} -8795771.21968346
"""
