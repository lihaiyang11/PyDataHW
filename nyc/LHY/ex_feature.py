import pandas as pd
import numpy as np
import get_pca_var as gpca
import wea_and_turn as wat
import hav_var as hv
#所有的需要提取的特征：
# 'speed_hvsn', 'speed_manhtn', 'pickup_pca0',
# 'pickup_pca1', 'dropoff_pca0', 'dropoff_pca1',
#  'store_and_fwd_flag_int', 'straight', 'left',
#  'right', 'minimum temperature', 'precipitation',
#  'snow fall', 'snow depth', 'hvsine_pick_cent_d',
#  'hvsine_drop_cent_p

def get_feature():
    dataframe = pd.read_csv('../data/train.csv')
    print("this is the csv read in \n")
    print(dataframe.head())
    print(dataframe.shape)
    # 得到相关的转弯，最快路径，总行驶时间，的信息，
    dataframe = wat.get_turn_number(dataframe)
    print(dataframe.head())

    # 加入天气相关信息，
    dataframe = wat.get_weather_info(dataframe)
    print(dataframe.head())
    
    # 加入节日信息
    dataframe = wat.get_holiday(dataframe)
    print(dataframe.head())
    
    # 加入变换后的坐标的信息，适用pca来变换坐标
    dataframe = gpca.get_pca(dataframe)
    print(dataframe.head())

    # 加入集群相关的球面距离，和球面距离，曼哈顿距离所对应的速度,需要用到travel_time
    #需要处理travel_time=0的异常值，除去对应行
    dataframe = dataframe[dataframe['total_travel_time'] != 0]
    dataframe = hv.ex_speed_hvsn(dataframe)
    dataframe = hv.ex_speed_manhtn(dataframe)
    print(dataframe.shape)
  #  dataframe.to_csv('result.csv')
    return dataframe
data = get_feature()
data.to_csv('../data/result.csv')





