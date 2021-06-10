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

train_data = pd.read_csv('../data/train.csv')
train_data['pickup_datetime'] = pd.to_datetime(train_data.pickup_datetime)
train_data.loc[:, 'pick_month'] = train_data['pickup_datetime'].dt.month
train_data.loc[:, 'hour'] = train_data['pickup_datetime'].dt.hour
train_data.loc[:, 'week_of_year'] = train_data['pickup_datetime'].dt.weekofyear
train_data.loc[:, 'day_of_year'] = train_data['pickup_datetime'].dt.dayofyear
train_data.loc[:, 'day_of_week'] = train_data['pickup_datetime'].dt.dayofweek
train_data.loc[:, 'hvsine_pick_drop'] = haversine_(train_data['pickup_latitude'].values,
                                                   train_data['pickup_longitude'].values,
                                                   train_data['dropoff_latitude'].values,
                                                   train_data['dropoff_longitude'].values)
train_data.loc[:, 'manhtn_pick_drop'] = manhattan_distance_pd(train_data['pickup_latitude'].values,
                                                              train_data['pickup_longitude'].values,
                                                              train_data['dropoff_latitude'].values,
                                                              train_data['dropoff_longitude'].values)
train_data.loc[:, 'bearing'] = bearing_array(train_data['pickup_latitude'].values,
                                             train_data['pickup_longitude'].values,
                                             train_data['dropoff_latitude'].values,
                                             train_data['dropoff_longitude'].values)

# 月份
summary_pmonth_avg_duration = pd.DataFrame(train_data.groupby(['pick_month'])['trip_duration'].mean())
summary_pmonth_avg_duration.reset_index(inplace=True)
summary_pmonth_avg_duration_avg_duration['trip_duration'].plot(kind='line', rot=0)
plt.show()

# 小时
summary_hour_avg_duration = pd.DataFrame(train_data.groupby(['hour'])['trip_duration'].mean())
summary_hour_avg_duration.reset_index(inplace=True)
summary_hour_avg_duration_avg_duration['trip_duration'].plot(kind='line', rot=0)
plt.show()

# 年的第几周
summary_wyear_avg_duration = pd.DataFrame(train_data.groupby(['week_of_year'])['trip_duration'].mean())
summary_wyear_avg_duration.reset_index(inplace=True)
summary_wyear_avg_duration['trip_duration'].plot(kind='line', rot=0)
plt.show()

# 年的第几天
summary_dyear_avg_duration = pd.DataFrame(train_data.groupby(['day_of_year'])['trip_duration'].mean())
summary_dyear_avg_duration.reset_index(inplace=True)
summary_dyear_avg_duration['trip_duration'].plot(kind='line', rot=0)
plt.show()

# 周几
summary_dweek_avg_duration = pd.DataFrame(train_data.groupby(['day_of_week'])['trip_duration'].mean())
summary_dweek_avg_duration.reset_index(inplace=True)
summary_dweek_avg_duration['trip_duration'].plot(kind='line', rot=0)
plt.show()

# hvsine距离
summary_hp2d_avg_duration = pd.DataFrame(train_data.groupby(['hvsine_pick_drop'])['trip_duration'].mean())
summary_hp2d_avg_duration.reset_index(inplace=True)
summary_hp2d_avg_duration['trip_duration'].plot(kind='line', rot=0)
plt.show()

# manhtn距离
summary_mp2d_avg_duration = pd.DataFrame(train_data.groupby(['manhtn_pick_drop'])['trip_duration'].mean())
summary_mp2d_avg_duration.reset_index(inplace=True)
summary_mp2d_avg_duration['trip_duration'].plot(kind='line', rot=0)
plt.show()

# bearing
summary_bearing_avg_duration = pd.DataFrame(train_data.groupby(['bearing'])['trip_duration'].mean())
summary_bearing_avg_duration.reset_index(inplace=True)
summary_bearing_avg_duration['trip_duration'].plot(kind='line', rot=0)
plt.show()

# 打印表格
summary_pmonth_avg_duration.to_csv("summary_pmonth_avg_duration.csv", index=False, sep=',')
summary_hour_avg_duration.to_csv("summary_hour_avg_duration.csv", index=False, sep=',')
summary_wyear_avg_duration.to_csv("summary_wyear_avg_duration.csv", index=False, sep=',')
summary_dyear_avg_duration.to_csv("summary_dyear_avg_duration.csv", index=False, sep=',')
summary_dweek_avg_duration.to_csv("summary_dweek_avg_duration.csv", index=False, sep=',')
summary_hp2d_avg_duration.to_csv("summary_hp2d_avg_duration.csv", index=False, sep=',')
summary_mp2d_avg_duration.to_csv("summary_mp2d_avg_duration.csv", index=False, sep=',')
summary_bearing_avg_duration.to_csv("summary_bearing_avg_duration.csv", index=False, sep=',')

# 将这八个特征输出到result文件内
train_data.to_csv('result.csv')
