import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

#pca是降维度的方法，将多个维度的特征降低，相当于综合经纬度给出一个特征。
#这里由于分布的平均性，导致变换维度之后仍旧有两个特征，可以看做是对
# 经纬度的一中变换。直观来说就是平面上将经纬度的坐标系转动
# 45度得到的新的“经纬度”
def get_pca(dataframe):
    coords = get_coords(dataframe)
    pca = PCA().fit(coords)
    dataframe['pickup_pca0'] = pca.transform(dataframe[['pickup_latitude','pickup_longitude']])[:,0]
    dataframe['pickup_pca1'] = pca.transform(dataframe[['pickup_latitude','pickup_longitude']])[:,1]
    dataframe['dropoff_pca0'] = pca.transform(dataframe[['dropoff_latitude', 'dropoff_longitude']])[:, 0]
    dataframe['dropoff_pca1'] = pca.transform(dataframe[['dropoff_latitude', 'dropoff_longitude']])[:, 1]
    return dataframe

def get_coords(dataframe):
    return np.vstack((dataframe[['pickup_latitude','pickup_longitude']].values,
                     dataframe[['dropoff_latitude','dropoff_longitude']].values,))