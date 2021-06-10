import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from bokeh.palettes import Spectral4
from bokeh.plotting import figure,output_notebook,show

#对于所有的变量给出其相关性的分布图
def draw_all_features():
    dataframe = pd.read_csv('result.csv')
    #给出相关性矩阵
    pearson = dataframe.corr()
    sns.heatmap(pearson)
    plt.show()
#给出pca得到的四个变量的范围分布,主要的分布区域:
# pickup_pca0 :[-0.8 ,0.8]
# pickup_pca1 :[-1.8, 2.2]
# dropoff_pca0:[-0.8 , 1.2]
# dropoff_pca1:[-2.3, 2.3]
def show_pca_var():
    dataframe = pd.read_csv('result.csv')
    sns.set(style="white", palette="muted", color_codes=True)
    f, axes = plt.subplots(2, 2, figsize=(10, 10), sharex=False, sharey=False)
    sns.despine(left=True)
    sns.distplot(dataframe['pickup_pca0'].values, label='pickup_pca0', color="m", bins=100, ax=axes[0, 0])
    sns.distplot(dataframe['pickup_pca1'].values, label='pickup_pca1', color="m", bins=100, ax=axes[0, 1])
    sns.distplot(dataframe['dropoff_pca0'].values, label='dropoff_pca0', color="m", bins=100, ax=axes[1, 0])
    sns.distplot(dataframe['dropoff_pca1'].values, label='dropoff_pca1', color="m", bins=100, ax=axes[1, 1])
    plt.setp(axes, yticks=[])
    plt.tight_layout()
    plt.show()
#表示duration在每个pca点上的散点图的分布。
def pca_with_duration():
    dataframe = pd.read_csv('result.csv')
    datanew = dataframe[dataframe['trip_duration'] > 60 ]
    datanew = datanew[datanew['trip_duration'] < 3600]
    #表示用pca0，pca1当做坐标，对应的点颜色代表duration的大小
    X = list(datanew['pickup_pca0'])
    Y = list(datanew['pickup_pca1'])
    Z = list(datanew['trip_duration'])
    plt.scatter(X,Y,s = 15,c = Z,alpha = .5)
    plt.xlim(-0.8,0.8)
    plt.ylim(-1.0,1.0)
    plt.show()

#展示每条数据的转弯数量和duration的关系
def show_turn_relation():
    data = pd.read_csv('result.csv')

    #取得对应每个转弯数量的duration的平均值，
    turn_left = pd.DataFrame(data.groupby('left')['trip_duration'].mean())
    turn_left.reset_index(inplace = True)
    turn_right = pd.DataFrame(data.groupby('right')['trip_duration'].mean())
    turn_right.reset_index(inplace= True)
    straight = pd.DataFrame(data.groupby('straight')['trip_duration'].mean())
    straight.reset_index(inplace= True)

    #依次画出三个变量的相关图
    turn_left['trip_duration'].plot(kind = 'line',rot = 0)
    plt.show()
    turn_right['trip_duration'].plot(kind='line', rot=0)
    plt.show()
    straight['trip_duration'].plot(kind='line', rot=0)
    plt.show()

#给出天气状况和对应的duration的关系
def show_weather():
    data = pd.read_csv('result.csv')

    #得到对应的天气状况下的duration的平均值
    min_temp = pd.DataFrame(data.groupby('minimum temperature')['trip_duration'].mean())
    min_temp.reset_index(inplace=True)
    snow_fall = pd.DataFrame(data.groupby('snow fall')['trip_duration'].mean())
    snow_fall.reset_index(inplace=True)
    snow_depth = pd.DataFrame(data.groupby('snow depth')['trip_duration'].mean())
    snow_depth.reset_index(inplace=True)
    precipitation = pd.DataFrame(data.groupby('precipitation')['trip_duration'].mean())
    precipitation.reset_index(inplace=True)

    #给出对应的图表
    min_temp['trip_duration'].plot(kind='line', rot=0)
    plt.show()
    snow_fall['trip_duration'].plot(kind='line', rot=0)
    plt.show()
    snow_depth['trip_duration'].plot(kind='line', rot=0)
    plt.show()
    precipitation['trip_duration'].plot(kind='line', rot=0)
    plt.show()

#给出对应的速度相关变量和duration的影响
def show_speed():
    data=pd.read_csv('result.csv')
    data['init1'] = 1000
    data['init2'] = 100
    data['speed_hvsn'] = data['speed_hvsn'] * data['init1']
    speed_hvsn = pd.DataFrame(data.groupby('speed_hvsn')['trip_duration'].mean())
    speed_hvsn.reset_index(inplace=True)
    data['speed_manhtn'] = data['speed_manhtn'] * data['init2']
    speed_manhtn = pd.DataFrame(data.groupby('speed_manhtn')['trip_duration'].mean())
    speed_manhtn.reset_index(inplace=True)

    speed_hvsn['trip_duration'].plot(kind='line', rot=0)
    plt.show()
    speed_manhtn['trip_duration'].plot(kind='line', rot=0)
    plt.show()
