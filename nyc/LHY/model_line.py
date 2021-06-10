import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cross_validation import train_test_split
from sklearn import metrics

data = pd.read_csv('result.csv')

#选取变量的对应列，可扩充
feature_cols = ['vendor_id','passenger_count','pickup_longitude','pickup_latitude',
                'dropoff_longitude','dropoff_latitude','store_and_fwd_flag','total_distance',
                'total_travel_time','straight','right','left','minimum temperature','snow fall',
                'snow depth','pickup_pca0','pickup_pca1','dropoff_pca0','dropoff_pca1',
                'havsine_pick_drop','speed_hvsn','manhtn_pick_drop','speed_manhtn']

print(data.shape)

#得到特征对应的矩阵，命名为X，
X = data[feature_cols]
X = pd.DataFrame(X,dtype=np.float)
print(X.shape)

#目标值对应的矩阵，对应于Y，
Y = data['trip_duration']
print(Y.shape)

#分割训练集和测试集
X_train = X.iloc[:1110000,:]
print(X_train.shape)
X_test = X.iloc[1150000:,:]
print(X_test.shape)
Y_train = Y.iloc[:1110000]
print(Y_train.shape)
Y_test = np.array(Y.iloc[1150000:])

#用X_train训练模型
linreg = LinearRegression()
linreg.fit(X_train,Y_train)
print('Now we get the model')

#利用模型预测
y_pred = np.array(linreg.predict(X_test))

#测试结果的评估得分
sum_pa = 0
for i in range(len(y_pred)):
    sum_pa += (np.log(y_pred[i] + 1) - np.log(Y_test[i] + 1) )**2
result = np.sqrt(sum_pa/len(y_pred))
print(result)
