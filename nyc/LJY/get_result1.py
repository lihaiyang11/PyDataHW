import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 计算已知经纬度的两点间距离
def haversine_(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371  # in km
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return (h)

# 计算已知经纬度的两点间距离

def manhattan_distance_pd(lat1, lng1, lat2, lng2):
    a = haversine_(lat1, lng1, lat1, lng2)
    b = haversine_(lat1, lng1, lat2, lng1)
    return a + b

# 在矩阵上工作

def bearing_array(lat1, lng1, lat2, lng2):
    """ function was taken from beluga's notebook as this function works on array
    while my function used to work on individual elements and was noticably slow"""
    AVG_EARTH_RADIUS = 6371  # in km
    lng_delta_rad = np.radians(lng2 - lng1)
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    return np.degrees(np.arctan2(y, x))

#提取特征
train_data = pd.read_csv('../data/train.csv')

train_data['pickup_datetime'] = pd.to_datetime(train_data.pickup_datetime)
train_data.loc[:, 'pick_month'] = train_data['pickup_datetime'].dt.month
train_data.loc[:, 'hour'] = train_data['pickup_datetime'].dt.hour
train_data.loc[:, 'week_of_year'] = train_data['pickup_datetime'].dt.weekofyear
train_data.loc[:, 'day_of_year'] = train_data['pickup_datetime'].dt.dayofyear
train_data.loc[:, 'day_of_week'] = train_data['pickup_datetime'].dt.dayofweek
train_data.loc[:, 'hvsine_pick_drop'] = haversine_(train_data['pickup_latitude'].values,train_data['pickup_longitude'].values,
                                                   train_data['dropoff_latitude'].values,train_data['dropoff_longitude'].values)

train_data.loc[:, 'manhtn_pick_drop'] = manhattan_distance_pd(train_data['pickup_latitude'].values,train_data['pickup_longitude'].values,
                                                              train_data['dropoff_latitude'].values,train_data['dropoff_longitude'].values)

train_data.loc[:, 'bearing'] = bearing_array(train_data['pickup_latitude'].values,train_data['pickup_longitude'].values,
                                             train_data['dropoff_latitude'].values,train_data['dropoff_longitude'].values)



# 将这八个特征输出到result文件内
print(train_data.head())
train_data.to_csv('../data/result.csv')
