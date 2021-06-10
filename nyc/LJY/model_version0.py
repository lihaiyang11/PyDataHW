import pandas as pd
import numpy as np
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
import lightgbm as lgb
import gc


train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')
#print(train.shape, test.shape)

#remove outliers  (about an hour and half max to go to JFK )将过大的数据删掉并输出最大时间和平均时间
train = train[train['trip_duration'] <  train['trip_duration'].quantile(0.999)]
train = train[train['trip_duration'] <= train['trip_duration'].mean() + 3*train['trip_duration'].std()]
print(' max time in min {:.2f} and mean time in sec {:.2f} '.format(
    train['trip_duration'].max()/60, train['trip_duration'].mean()))

# NYC longitude and latitude borders 处理经纬度 删掉过于偏僻的地点
ep = 0.0001
(lng1,lng2)=(-74.257*(1+ep), -73.699*(1-ep))
(lat1,lat2)=(40.495*(1+ep), 40.915*(1-ep))
train = train[(train['pickup_longitude'] <=lng2)&(train['pickup_longitude'] >=lng1)]
train = train[(train['pickup_latitude'] <=lat2) & (train['pickup_latitude'] >=lat1)]
train = train[(train['dropoff_longitude'] <=lng2)&(train['dropoff_longitude'] >=lng1)]
train = train[(train['dropoff_latitude'] <=lat2)&(train['dropoff_latitude'] >=lat1)]

#  combine train and test into a single frame 将train和test两个数据集合到一起，并分别给一个评估值
train['eval_set'] = 1
test['eval_set'] = 2
Data = pd.concat([train, test], axis=0,sort=False) #因为报错所以添加sort=False
#print(train.shape,test.shape,len(test)+len(train),Data.shape)

del train, test #合并后就可以删掉了
gc.collect()#内存释放
#Data.head(2)

# transform the time variables, combine hours and minutes and add holidays 求旅行时间的对数、处理细化上车时间并处理是否回复出租车公司这个变量。删掉指定的三个标签
Data['log_trip_duration'] = np.log(Data['trip_duration'].values + 1)
Data['pickup_datetime'] = pd.to_datetime(Data.pickup_datetime)
Data['Month'] = Data['pickup_datetime'].dt.month
Data['DayofMonth'] = Data['pickup_datetime'].dt.day
Data['Timehm'] =Data['pickup_datetime'].dt.hour+Data['pickup_datetime'].dt.minute/60.
Data['dayofweek'] = Data['pickup_datetime'].dt.dayofweek
Data['dayofyear'] = Data['pickup_datetime'].dt.dayofyear
Data['weekofyear'] = Data['pickup_datetime'].dt.weekofyear
holidays = calendar().holidays(start='2015-12-31', end='2016-07-01')
Data['Holiday']=Data['pickup_datetime'].dt.date.astype('datetime64[ns]').isin(holidays).astype(int)
Data.drop(['trip_duration','dropoff_datetime','pickup_datetime'], axis=1,inplace=True)
# and replace the N,Y with 0,1
Data['store_and_fwd_flag'] = Data['store_and_fwd_flag'].map({'N': 0, 'Y':1}).astype(int)
#Data.shape

def haversine_array(lat1, lng1, lat2, lng2):#计算距离 ×× power
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    d = np.sin(lat2/2-lat1/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng2/2-lng1/2)**2
    return 2 * 6371 * np.arcsin(np.sqrt(d))   # 6,371 km is the earth radius

def dummy_manhattan_distance(lat1, lng1, lat2, lng2):#计算距离
    return haversine_array(lat1,lng1,lat1,lng2)+haversine_array(lat1,lng1,lat2,lng1)

def bearing_array(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    y = np.sin(lng2 - lng1) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng2 - lng1)
    return np.degrees(np.arctan2(y, x))

Data['distance_haversine'] = haversine_array(Data['pickup_latitude'].values, Data['pickup_longitude'].values,
                                          Data['dropoff_latitude'].values, Data['dropoff_longitude'].values)
Data['distance_dummy_manhattan'] =  dummy_manhattan_distance(Data['pickup_latitude'].values,
            Data['pickup_longitude'].values, Data['dropoff_latitude'].values, Data['dropoff_longitude'].values)
Data['direction'] = bearing_array(Data['pickup_latitude'].values, Data['pickup_longitude'].values,
                                      Data['dropoff_latitude'].values, Data['dropoff_longitude'].values)
#Data.shape

#  replace latitudes and longitudes with 81 clusters 对经纬度做集群处理

coords = np.vstack((Data[['pickup_latitude', 'pickup_longitude']].values,
                    Data[['dropoff_latitude', 'dropoff_longitude']].values))
sample_ind = np.random.permutation(len(coords))[:1000000]
kmeans = MiniBatchKMeans(n_clusters=9**2, batch_size=36**3).fit(coords[sample_ind])#样本量大于一万时使用的聚类算法
#n_clusters: 即我们的k值
# batch_size：即用来跑Mini Batch KMeans算法的采样集的大小，默认是100.如果发现数据集的类别较多或者噪音点较多，需要增加这个值以达到较好的聚类效果
#fit聚类内容拟合
Data['pickup_cluster'] = kmeans.predict(Data[['pickup_latitude', 'pickup_longitude']])#聚类预测
Data['dropoff_cluster'] = kmeans.predict(Data[['dropoff_latitude', 'dropoff_longitude']])#聚类预测
#Data.shape

pca = PCA().fit(coords)
Data['pickup_pca0'] = pca.transform(Data[['pickup_latitude', 'pickup_longitude']])[:, 0]
Data['pickup_pca1'] = pca.transform(Data[['pickup_latitude', 'pickup_longitude']])[:, 1]
Data['dropoff_pca0'] = pca.transform(Data[['dropoff_latitude', 'dropoff_longitude']])[:, 0]
Data['dropoff_pca1'] = pca.transform(Data[['dropoff_latitude', 'dropoff_longitude']])[:, 1]
Data['pca_manhattan'] = np.abs(Data['dropoff_pca1']-Data['pickup_pca1']) +  np.abs(
                                 Data['dropoff_pca0']-Data['pickup_pca0'])
Data['center_latitude'] = 0.5*Data['pickup_latitude']+0.5*Data['dropoff_latitude']
Data['center_longitude'] = 0.5*Data['pickup_longitude']+0.5*Data['dropoff_longitude']
del coords, sample_ind, kmeans
#Data.shape

#处理路线
flag=False
if flag:
    cols=['id', 'total_distance', 'total_travel_time',  'number_of_steps']
    #usecols选取指定列 处理后加入Data
    fr1 = pd.read_csv('/home/apple/Downloads/fastest_routes_train_part_1.csv', usecols=cols)
    fr2 = pd.read_csv('/home/apple/Downloads/fastest_routes_train_part_2.csv', usecols=cols)
    fr3 = pd.read_csv('/home/apple/Downloads/fastest_routes_test.csv', usecols=cols)
    tmp = pd.concat((fr1, fr2, fr3))
    tmp.columns=['id', 'OSRM_distance', 'OSRM_time',  'OSRM_steps']
    Data = Data.merge(tmp, how='left', on='id')
    del fr1, fr2, fr3, tmp
    gc.collect()
    #Data.shape
print(flag)

# split the data set back to train and test (for submission) and the train into "train" and "eval" sets for ML 分回train和test
# save the ids for future

Test_id = Data['id'].loc[Data['eval_set']==2].to_frame()
Data.drop('id',axis = 1, inplace=True)
train = Data[Data['eval_set'] == 1]
test = Data[Data['eval_set'] == 2]
print(Data.shape, train.shape, test.shape)

X=train.drop(['eval_set','log_trip_duration'],axis=1)
y=train['log_trip_duration']
features=list(X.columns)
cfeatures = list(X.select_dtypes(include = ['int64','int32']).columns)

X_train, X_eval, y_train, y_eval = train_test_split(X,y, test_size=0.2, random_state=2)

del train, Data
gc.collect()

print('formatting and training LightGBM regression ...')
# use higher num_boost_round eg 1000

lgb_train = lgb.Dataset(X_train.values, y_train.values)
lgb_eval = lgb.Dataset(X_eval.values, y_eval.values, reference = lgb_train)
#metric 损失 learningrate 学习率 numleaves 一棵树上的叶子数
#feature_fraction 每次迭代随机选择90%的数据
#bagging_fraction 随机选择90%数据执行bagging
#bagging_freq 数据子样本每五次迭代执行一次bagging
#min_data_in_leaf 一个叶子的最小数据, 用它来处理过度over-fit
params = {'metric': 'rmse', 'learning_rate' : 0.05, 'num_leaves': 512,
         'feature_fraction': 0.9,'bagging_fraction':0.9,'bagging_freq':5,'min_data_in_leaf': 500}
#num_boost_round提升迭代的个数
#valid_sets 测试数据
#early_stopping_rounds 早期停止次数 ，假设为100，验证集的误差迭代到一定程度在100次内不能再继续降低，就停止迭代。
#verbose_eval (可以输入布尔型或数值型)，也要求evals 里至少有 一个元素。如果为True ,则对evals中元素的评估结果会输出在结果中；
                # 如果输入数字，假设为5，则每隔5个迭代输出一次。
lgb_model = lgb.train(params, lgb_train, num_boost_round = 200, valid_sets = lgb_eval,
             feature_name=features, early_stopping_rounds=10,  verbose_eval = 10)

del lgb_train
gc.collect()

#num_iteration 迭代次数
#如果early_stopping_rounds 存在，则模型会生成三个属性，bst.best_score,bst.best_iteration（最好的迭代次数）,和bst.best_ntree_limit
pred = lgb_model.predict(test.drop(['eval_set','log_trip_duration'], axis=1).values,
                         num_iteration = lgb_model.best_iteration)
Test_id['trip_duration']=np.maximum(0,np.exp(pred) - 1)
Test_id.to_csv('submission.csv', index=False)#两列 一列id 一列时间
print(Test_id['trip_duration'].mean())           # just a sanity check - it should be around 806估计一个测试时间的平均值


#check the score again
pred1 = lgb_model.predict(X_train.values, num_iteration = lgb_model.best_iteration)
pred2 = lgb_model.predict(X_eval.values, num_iteration = lgb_model.best_iteration)
rmsle1= (((y_train-pred1)**2).mean())**0.5
rmsle2 = (((y_eval-pred2)**2).mean())**0.5
print('train score: {:.4f}   eval score: {:.4f}'.format(rmsle1,rmsle2))

