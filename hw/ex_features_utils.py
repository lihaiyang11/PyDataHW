import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def get_coords(dataframe):
    return np.vstack((dataframe[['pickup_latitude', 'pickup_longitude']].values,
                      dataframe[['dropoff_latitude', 'dropoff_longitude']].values,))


# 集群相关的数据，分别表示的直观数据是
# pick点到drop集群中心，drop点到pick集群中心。
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


# 辅助函数，用于求解球面距离和manhtn距离，球面距离中，lat1,
# lat2,等分别为对应经纬度对地心的张角，加上半径的数据就能够计算
# 两个点在球面的距离
def haversine_(lat1, lng1, lat2, lng2):
    """function to calculate haversine distance between two co-ordinates"""
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371  # in km
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return (h)


# manhtn距离表示如下：是计算两个地点的经纬度差所带来的直角两条边距离之和
def manhattan_distance_pd(lat1, lng1, lat2, lng2):
    """function to calculate manhatten distance between pick_drop"""
    a = haversine_(lat1, lng1, lat1, lng2)
    b = haversine_(lat1, lng1, lat2, lng1)
    return a + b


# 分割字符串来计算每个路径中的r，l，s的数量，step_dir形如"left|right"的字符串
# a_list是一个键值对数组，形如{(left,2)(right,3)}
# 后面对a遍历得到每个step_dir中left，r，s的数量并返回。
def freq_turn(step_dir):
    from collections import Counter
    step_dir_new = step_dir.split("|")
    a_list = Counter(step_dir_new).most_common()
    path = {}
    for i in range(len(a_list)):
        path.update({a_list[i]})
    a = 0
    b = 0
    c = 0
    if 'straight' in (path.keys()):
        a = path['straight']
        # print(a)
    if 'left' in (path.keys()):
        b = path['left']
        # print(b)
    if 'right' in (path.keys()):
        c = path['right']
        # print(c)
    return a, b, c


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


# 集群生成函数，k为集群的数量，函数返回处理之后的数据集与集群
def assign_cluster(df, k):
    df_pick = df[['pickup_longitude', 'pickup_latitude']]
    df_drop = df[['dropoff_longitude', 'dropoff_latitude']]
    init = np.array([
        [-73.98737616, 40.72981533],
        [-121.93328857, 37.38933945],
        [-73.78423222, 40.64711269],
        [-73.9546417, 40.77377538],
        [-66.84140269, 36.64537175],
        [-73.87040541, 40.77016484],
        [-73.97316185, 40.75814346],
        [-73.98861094, 40.7527791],
        [-72.80966949, 51.88108444],
        [-76.99779701, 38.47370625],
        [-73.96975298, 40.69089596],
        [-74.00816622, 40.71414939],
        [-66.97216034, 44.37194443],
        [-61.33552933, 37.85105133],
        [-73.98001393, 40.7783577],
        [-72.00626526, 43.20296402],
        [-73.07618713, 35.03469086],
        [-73.95759366, 40.80316361],
        [-79.20167796, 41.04752096],
        [-74.00106031, 40.73867723]])
    k_means_pick = KMeans(n_clusters=k, init=init, n_init=1)
    k_means_pick.fit(df_pick)
    clust_pick = k_means_pick.labels_
    # 对每条数据的pick与drop位置备注上对应的集群编号
    df['label_pick'] = clust_pick.tolist()
    df['label_drop'] = k_means_pick.predict(df_drop)
    return df, k_means_pick


# 直行，左转右转的数量，
def get_turn_number(dataframe):
    # datafr1 = pd.read_csv("../data/fastest_routes_train_part_1.csv")
    # datafr2 = pd.read_csv("../data/fastest_routes_train_part_2.csv")
    # datafr = pd.concat([datafr1, datafr2])
    #
    # datafr['straight'] = 0
    # datafr['right'] = 0
    # datafr['left'] = 0
    # datafr['straight'], datafr['left'], datafr['right'] = zip(*datafr['step_direction'].map(freq_turn))
    #
    # data_new = datafr[['id','straight', 'right', 'left']]
    # dataframe = pd.merge(dataframe, data_new, on='id', how='left')
    dataframe['straight'] = 0
    dataframe['right'] = 0
    dataframe['left'] = 0
    dataframe['straight'], dataframe['left'], dataframe['right'] = zip(*dataframe['step_direction'].map(freq_turn))
    return dataframe


# 天气信息，最低温度和降水量,降雪量和降雪厚度
def get_weather_info(dataframe):
    weather = pd.read_csv("../data/weather_data.csv")

    # 这里需要对于表格中的bool类型的数据处理成为浮点型变量
    weather['date'] = pd.to_datetime(weather['date'])
    weather['precipitation'].unique()
    weather['precipitation'] = np.where(weather['precipitation'] == 'T', '0.00', weather['precipitation'])
    weather['precipitation'] = list(map(float, weather['precipitation']))
    weather['snow fall'] = np.where(weather['snow fall'] == 'T', '0.00', weather['snow fall'])
    weather['snow fall'] = list(map(float, weather['snow fall']))
    weather['snow depth'] = np.where(weather['snow depth'] == 'T', '0.00', weather['snow depth'])
    weather['snow depth'] = list(map(float, weather['snow depth']))

    dataframe['pickup_datetime'] = pd.to_datetime(dataframe['pickup_datetime'])
    dataframe['date'] = dataframe['pickup_datetime'].dt.date
    dataframe['date'] = pd.to_datetime(dataframe['date'])
    dataframe = pd.merge(dataframe, weather[['date', 'minimum temperature',
                                             'precipitation', 'snow fall',
                                             'snow depth']], on='date', how='left')
    return dataframe


def get_holiday(dataframe):
    holiday = pd.read_csv('../data/holiday.csv')

    holiday['date'] = pd.to_datetime(holiday['date'])
    holiday['is_holiday'] = list(map(float, holiday['is_holiday']))
    dataframe['pickup_datetime'] = pd.to_datetime(dataframe['pickup_datetime'])
    dataframe['date'] = dataframe['pickup_datetime'].dt.date
    dataframe['date'] = pd.to_datetime(dataframe['date'])
    dataframe = pd.merge(dataframe, holiday[['date', 'is_holiday']], on='date', how='left')
    return dataframe


# pca是降维度的方法，将多个维度的特征降低，相当于综合经纬度给出一个特征。
# 这里由于分布的平均性，导致变换维度之后仍旧有两个特征，可以看做是对
# 经纬度的一中变换。直观来说就是平面上将经纬度的坐标系转动
# 45度得到的新的“经纬度”
def get_pca(dataframe):
    coords = get_coords(dataframe)
    pca = PCA().fit(coords)
    dataframe['pickup_pca0'] = pca.transform(dataframe[['pickup_latitude', 'pickup_longitude']])[:, 0]
    dataframe['pickup_pca1'] = pca.transform(dataframe[['pickup_latitude', 'pickup_longitude']])[:, 1]
    dataframe['dropoff_pca0'] = pca.transform(dataframe[['dropoff_latitude', 'dropoff_longitude']])[:, 0]
    dataframe['dropoff_pca1'] = pca.transform(dataframe[['dropoff_latitude', 'dropoff_longitude']])[:, 1]
    return dataframe


# speed_hvsn = havsine_pick_drop / travel_time;
def ex_speed_hvsn(dataframe):
    dataframe['havsine_pick_drop'] = haversine_(dataframe['pickup_latitude'],
                                                dataframe['pickup_longitude'],
                                                dataframe['dropoff_latitude'],
                                                dataframe['dropoff_longitude'])
    dataframe['speed_hvsn'] = dataframe['havsine_pick_drop'].values / dataframe['total_travel_time'].values
    return dataframe


# speed_manhtn = manhtn_pick_drop / travel_time
def ex_speed_manhtn(dataframe):
    dataframe['manhtn_pick_drop'] = manhattan_distance_pd(dataframe['pickup_latitude'],
                                                          dataframe['pickup_longitude'],
                                                          dataframe['dropoff_latitude'],
                                                          dataframe['dropoff_longitude'])
    dataframe['speed_manhtn'] = dataframe['manhtn_pick_drop'].values / dataframe['total_travel_time'].values
    return dataframe


def get_dateinfo(dataframe):
    dataframe['pickup_datetime'] = pd.to_datetime(dataframe.pickup_datetime)
    dataframe.loc[:, 'pick_month'] = dataframe['pickup_datetime'].dt.month
    dataframe.loc[:, 'hour'] = dataframe['pickup_datetime'].dt.hour
    dataframe.loc[:, 'week_of_year'] = dataframe['pickup_datetime'].dt.weekofyear
    dataframe.loc[:, 'day_of_year'] = dataframe['pickup_datetime'].dt.dayofyear
    dataframe.loc[:, 'day_of_week'] = dataframe['pickup_datetime'].dt.dayofweek
    return dataframe


def get_cluster(dataframe):
    train_cl, k_means = assign_cluster(dataframe, 20)
    centroid_pickups = pd.DataFrame(k_means.cluster_centers_, columns=['centroid_pick_long', 'centroid_pick_lat'])
    centroid_dropoff = pd.DataFrame(k_means.cluster_centers_, columns=['centroid_drop_long', 'centroid_drop_lat'])
    centroid_pickups['label_pick'] = centroid_pickups.index
    centroid_dropoff['label_drop'] = centroid_dropoff.index
    train_cl = pd.merge(train_cl, centroid_pickups, how='left', on=['label_pick'])
    train_cl = pd.merge(train_cl, centroid_dropoff, how='left', on=['label_drop'])
    return train_cl


def get_havsine_info(dataframe):
    dataframe.loc[:, 'hvsine_pick_cent_p'] = haversine_(dataframe['pickup_latitude'].values,
                                                        dataframe['pickup_longitude'].values,
                                                        dataframe['centroid_pick_lat'].values,
                                                        dataframe['centroid_pick_long'].values)
    dataframe.loc[:, 'hvsine_drop_cent_d'] = haversine_(dataframe['dropoff_latitude'].values,
                                                        dataframe['dropoff_longitude'].values,
                                                        dataframe['centroid_drop_lat'].values,
                                                        dataframe['centroid_drop_long'].values)
    dataframe.loc[:, 'hvsine_cent_p_cent_d'] = haversine_(dataframe['centroid_pick_lat'].values,
                                                          dataframe['centroid_pick_long'].values,
                                                          dataframe['centroid_drop_lat'].values,
                                                          dataframe['centroid_drop_long'].values)
    dataframe['havsine_pick_drop'] = haversine_(dataframe['pickup_latitude'],
                                                dataframe['pickup_longitude'],
                                                dataframe['dropoff_latitude'],
                                                dataframe['dropoff_longitude'])
    # speed_hvsn = havsine_pick_drop / travel_time;
    dataframe['speed_hvsn'] = dataframe['havsine_pick_drop'].values / dataframe['total_travel_time'].values
    # pick点到drop集群中心，drop点到pick集群中心。
    dataframe['hvsine_pick_cent_d'] = haversine_(dataframe['pickup_latitude'].values,
                                                 dataframe['pickup_longitude'].values,
                                                 dataframe['centroid_drop_lat'].values,
                                                 dataframe['centroid_drop_long'].values)
    dataframe['hvsine_drop_cent_p'] = haversine_(dataframe['dropoff_latitude'].values,
                                                 dataframe['dropoff_longitude'].values,
                                                 dataframe['centroid_pick_lat'].values,
                                                 dataframe['centroid_pick_long'].values)
    return dataframe


def get_manhatn_info(dataframe):
    dataframe.loc[:, 'manhtn_pick_cent_p'] = manhattan_distance_pd(dataframe['pickup_latitude'].values,
                                                                   dataframe['pickup_longitude'].values,
                                                                   dataframe['centroid_pick_lat'].values,
                                                                   dataframe['centroid_pick_long'].values)
    dataframe.loc[:, 'manhtn_drop_cent_d'] = manhattan_distance_pd(dataframe['dropoff_latitude'].values,
                                                                   dataframe['dropoff_longitude'].values,
                                                                   dataframe['centroid_drop_lat'].values,
                                                                   dataframe['centroid_drop_long'].values)
    dataframe.loc[:, 'manhtn_cent_p_cent_d'] = manhattan_distance_pd(dataframe['centroid_pick_lat'].values,
                                                                     dataframe['centroid_pick_long'].values,
                                                                     dataframe['centroid_drop_lat'].values,
                                                                     dataframe['centroid_drop_long'].values)
    dataframe['manhtn_pick_drop'] = manhattan_distance_pd(dataframe['pickup_latitude'],
                                                          dataframe['pickup_longitude'],
                                                          dataframe['dropoff_latitude'],
                                                          dataframe['dropoff_longitude'])
    # speed_manhtn = manhtn_pick_drop / travel_time
    dataframe['speed_manhtn'] = dataframe['manhtn_pick_drop'].values / dataframe['total_travel_time'].values
    return dataframe
