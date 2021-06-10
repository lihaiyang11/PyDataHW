import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

data = pd.read_csv('../data/result.csv')

#选取变量的对应列，可扩充
feature_cols = ['vendor_id','passenger_count','pickup_longitude','pickup_latitude',
                'dropoff_longitude','dropoff_latitude','store_and_fwd_flag_int','total_distance',
                'total_travel_time','straight','right','left',
                'minimum temperature','snow fall','snow depth','pickup_pca0',
                'pickup_pca1','dropoff_pca0','dropoff_pca1','havsine_pick_drop',
                'speed_hvsn','manhtn_pick_drop','speed_manhtn','bearing',
                'label_pick','label_drop','centroid_pick_long','centroid_pick_lat',
                'centroid_drop_long','centroid_drop_lat','hvsine_pick_cent_p','hvsine_drop_cent_d',
                'hvsine_pick_cent_d','hvsine_drop_cent_p','hvsine_cent_p_cent_d','manhtn_pick_cent_p',
                'manhtn_drop_cent_d','manhtn_cent_p_cent_d','bearing_pick_cent_p','bearing_drop_cent_p',
                'bearing_cent_p_cent_d','pick_month','hour','week_of_year',
                'day_of_year','day_of_week','is_holiday','speed_manhtn',
                'speed_hvsn'
                ]
category_cols = ['vendor_id','passenger_count','store_and_fwd_flag_int',
                 'pick_month','day_of_week','is_holiday','hour']

"""
给出数据清理的过程，只有第一次运行时需要，第一次运行后会吧数据存起来，
for i in feature_cols:
    data = data[pd.notnull(data[i])]
    print(i)
    print(data.shape)
data.to_csv('../data/result.csv')
"""

print(data.shape)
#转换类别为整数
for col in category_cols:
    data[col] = data[col].astype('category').cat.codes
#得到特征对应的矩阵，命名为X，
X = data[feature_cols]
#X = pd.DataFrame(X,dtype=np.float)
print(X.shape)

cat_feature = [X.columns.get_loc(i) for i in category_cols[:-1]]
#目标值对应的矩阵，对应于Y，
Y = data['trip_duration']
print(Y.shape)

#分割训练集和测试集
X_train = X.iloc[:1200000,:]
print(X_train.shape)
X_test = X.iloc[1200000:,:]
print(X_test.shape)
Y_train = Y.iloc[:1200000]
print(Y_train.shape)
Y_test = np.array(Y.iloc[1200000:])

rate = [0.57,0.55,0.53,0.51,0.49,0.59]
depth = [3,1,2,6,4,5,7,8,9,10]
iterations = [4,2]
l2_leaf_reg = [100,200,300]
border_count = [32,5,10,20,50,100,200]
ctr_border_count = [50,5,10,20,100,200]

for i in depth:
    #建立模型，参数可调
    model = CatBoostRegressor(iterations=2,depth=7,l2_leaf_reg=200,
                              learning_rate=0.51,loss_function='RMSE',
                              logging_level='Verbose',od_pval=0.000001,
                              border_count=20)

    #模型训练，可以分两种训练模式

    #不规定分类变量的训练
    #model.fit(X_train,y=Y_train)

    #规定类别的训练
    model.fit(X_train,y=Y_train,cat_features=cat_feature,plot=True)


    #训练的模型预测
    y_pred = np.array(model.predict(X_test))
    y_pred[y_pred < 0] = 0

    #测试结果的评估得分
    sum_pa = 0
    for i in range(len(y_pred)):
        sum_pa += (np.log(y_pred[i] + 1) - np.log(Y_test[i] + 1) )**2
    result = np.sqrt(sum_pa/len(y_pred))
    print(result)


