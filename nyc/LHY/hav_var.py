import numpy as np
import pandas as pd


#集群相关的数据，分别表示的直观数据是
#pick点到drop集群中心，drop点到pick集群中心。
def ex_hvsine_pick_cent_d(dataframe):
    dataframe['hvsine_pick_cent_d'] = haversine_(dataframe['pickup_latitude'].values,
                                                 dataframe['pickup_longitude'].values,
                                                 dataframe['centroid_drop_lat'].values,
                                                 dataframe['centroid_drop_long'].values)
    return dataframe


def ex_hvsine_drop_cent_p(dataframe):
    dataframe['hvsine_drop_cent_p'] = haversine_(dataframe['dropoff_latitude'].values,
                                                 dataframe['dropoff_longitude'].values,
                                                 dataframe['centroid_pick_lat'].values,
                                                 dataframe['centroid_pick_long'].values)
    return dataframe


# speed_hvsn = havsine_pick_drop / travel_time;
def ex_speed_hvsn(dataframe):
    dataframe['havsine_pick_drop'] = haversine_(dataframe['pickup_latitude'],
                                                dataframe['pickup_longitude'],
                                                dataframe['dropoff_latitude'],
                                                dataframe['dropoff_longitude'])
    dataframe['speed_hvsn'] = dataframe['havsine_pick_drop'].values/dataframe['total_travel_time'].values
    return dataframe

# speed_manhtn = manhtn_pick_drop / travel_time
def ex_speed_manhtn(dataframe):
    dataframe['manhtn_pick_drop'] = manhattan_distance_pd(dataframe['pickup_latitude'],
                                                          dataframe['pickup_longitude'],
                                                          dataframe['dropoff_latitude'],
                                                          dataframe['dropoff_longitude'])
    dataframe['speed_manhtn'] = dataframe['manhtn_pick_drop'].values/dataframe['total_travel_time'].values
    return dataframe
#辅助函数，用于求解球面距离和manhtn距离，球面距离中，lat1,
#lat2,等分别为对应经纬度对地心的张角，加上半径的数据就能够计算
#两个点在球面的距离
def haversine_(lat1, lng1, lat2, lng2):
    """function to calculate haversine distance between two co-ordinates"""
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371  # in km
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return(h)
#manhtn距离表示如下：是计算两个地点的经纬度差所带来的球面距离之和
def manhattan_distance_pd(lat1, lng1, lat2, lng2):
    """function to calculate manhatten distance between pick_drop"""
    a = haversine_(lat1, lng1, lat1, lng2)
    b = haversine_(lat1, lng1, lat2, lng1)
    return a + b

