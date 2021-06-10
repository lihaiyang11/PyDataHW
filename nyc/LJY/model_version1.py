import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import gc


data = pd.read_csv('/home/apple/NYC-Taxi/first/result.csv')

# NYC longitude and latitude borders 处理经纬度 删掉过于偏僻的地点
data = data[data['trip_duration'] <  data['trip_duration'].quantile(0.999)]
data = data[data['trip_duration'] <= data['trip_duration'].mean() + 3*data['trip_duration'].std()]
ep = 0.0001
(lng1,lng2)=(-74.257*(1+ep), -73.699*(1-ep))
(lat1,lat2)=(40.495*(1+ep), 40.915*(1-ep))
data = data[(data['pickup_longitude'] <=lng2)&(data['pickup_longitude'] >=lng1)]
data = data[(data['pickup_latitude'] <=lat2) & (data['pickup_latitude'] >=lat1)]
data = data[(data['dropoff_longitude'] <=lng2)&(data['dropoff_longitude'] >=lng1)]
data = data[(data['dropoff_latitude'] <=lat2)&(data['dropoff_latitude'] >=lat1)]

#选取变量的对应列，可扩充
feature_cols = ['vendor_id','passenger_count','pickup_longitude',
                'pickup_latitude','dropoff_longitude','dropoff_latitude',
                'trip_duration','total_distance','total_travel_time',
                'number_of_steps','pick_month','hour','week_of_year','day_of_year',
                'day_of_week','hvsine_pick_drop','manhtn_pick_drop','bearing',
                'pickup_pca0','pickup_pca1','dropoff_pca0','dropoff_pca1',
                'havsine_pick_drop','speed_hvsn','speed_manhtn','label_pick','label_drop',
                'centroid_pick_long','centroid_pick_lat','centroid_drop_long',
                'centroid_drop_lat','hvsine_pick_cent_p','hvsine_drop_cent_d',
                'hvsine_cent_p_cent_d','manhtn_pick_cent_p','manhtn_drop_cent_d',
                'manhtn_cent_p_cent_d','bearing_pick_cent_p','bearing_drop_cent_p',
                'bearing_cent_p_cent_d']
#print(data.shape)

#train
y = np.log(data['trip_duration'].values + 1)
X_train, X_eval, y_train, y_eval = train_test_split(data[feature_cols].values , y , test_size=0.2 , random_state=2018)

lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_eval, y_eval, reference = lgb_train)

params = {'metric': 'rmse',
          'learning_rate' :1.0,   #试过0.02
          'num_leaves': 1000,     #xgb为10 比2^10略小
          'feature_fraction': 0.3,#feature_fraction 每次迭代随机选择30%的数据
          'bagging_fraction':0.8, #bagging_fraction 随机选择80%数据执行bagging
          'bagging_freq':5,
          'min_data_in_leaf': 200} #min_data_in_leaf 一个叶子的最小数据, 用它来处理过度over-fit
#num_boost_round提升迭代的个数
#valid_sets 测试数据
#early_stopping_rounds 早期停止次数 ，假设为100，验证集的误差迭代到一定程度在100次内不能再继续降低，就停止迭代。
#verbose_eval (可以输入布尔型或数值型)，也要求evals 里至少有 一个元素。如果为True ,则对evals中元素的评估结果会输出在结果中；
lgb_model = lgb.train(params,
                      lgb_train,
                      num_boost_round = 200,#num_boost_round提升迭代的个数
                      valid_sets = lgb_eval,#valid_sets 测试数据
                      feature_name=feature_cols,
                      early_stopping_rounds=50,#early_stopping_rounds 早期停止次数 ，
                                               #假设为50，验证集的误差迭代到一定程度在50次内不能再继续降低，就停止迭代。
                      verbose_eval = 10)#verbose_eval (可以输入布尔型或数值型)，也要求evals 里至少有一个元素。
                                        #如果为数字 假设为10，则每隔10个迭代输出一次。
del lgb_train
gc.collect()

#check score
pred1 = lgb_model.predict(X_train, num_iteration = lgb_model.best_iteration)
pred2 = lgb_model.predict(X_eval, num_iteration = lgb_model.best_iteration)
rmsle1= (((y_train-pred1)**2).mean())**0.5
rmsle2 = (((y_eval-pred2)**2).mean())**0.5
print('train score: {:.4f}   eval score: {:.4f}'.format(rmsle1,rmsle2))


