import numpy as np
import pandas as pd
import math
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

data = pd.read_csv('../data/result.csv')
feature_names = ['vendor_id','passenger_count','pickup_longitude',
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

y = np.log(data['trip_duration'].values + 1)
Xtr, Xv, ytr, yv = train_test_split(data[feature_names].values, y, test_size = 0.2, random_state = 1987)
dtrain = xgb.DMatrix(Xtr, label = ytr)
dvalid = xgb.DMatrix(Xv, label = yv)
dtest = xgb.DMatrix(data[feature_names].values)
watchlist = [(dtrain, 'train'), (dvalid, 'valid')]

xgb_pars = {'min_child_weight':50, 'eta':0.3, 'colsample_bytree': 0.3, 'max_depth':10,
            'subsample':0.8, 'lamba': 1, 'ntherad':-1, 'booster': 'gbtree', 'silent':1,
            'eval_metric':'rmse', 'objective':'reg:linear'}
model = xgb.train(xgb_pars, dtrain, 15, watchlist, early_stopping_rounds = 2, maximize = False, verbose_eval = 1)
print('Modeling RMSLE %.5f' % model.best_score)

