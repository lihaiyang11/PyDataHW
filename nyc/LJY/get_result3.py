#李海洋 不包含天气和行驶路径的 特征值
import pandas as pd
import get_pca_var as gpca
import hav_var as hv

def get_feature():
    dataframe = pd.read_csv('../data/result.csv')
    print("this is the csv read in \n")
    print(dataframe.head())
    print(dataframe.shape)

    # 加入变换后的坐标的信息，适用pca来变换坐标
    dataframe = gpca.get_pca(dataframe)
    print(dataframe.head())

    # 加入集群相关的球面距离，和球面距离，曼哈顿距离所对应的速度,需要用到travel_time
    #需要处理travel_time=0的异常值，除去对应行
    dataframe = dataframe[dataframe['total_travel_time'] != 0]
    dataframe = hv.ex_speed_hvsn(dataframe)
    dataframe = hv.ex_speed_manhtn(dataframe)
    print(dataframe.shape)
    dataframe.to_csv('result.csv')
    return dataframe
data = get_feature()
data.to_csv('result.csv')
