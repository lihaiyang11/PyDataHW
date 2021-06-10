# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 21:10:43 2018

@author: dell
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from bokeh.palettes import Spectral4
from bokeh.plotting import figure,output_notebook,show

#所有特征之间的pearson相关系数，用热点图表示
def draw_all_features():
    data = pd.read_csv('result.csv')
    pearson = data.corr()
    sns.heatmap(pearson)
    plt.show()
    return

#不同类型的方位与duration之间的关系
def show_bearing():
    data = pd.read_csv('result.csv')
    sns.regplot(x = 'bearing', y = 'trip_duration', data = data)
    plt.show()
    
    sns.regplot(x = 'bearing_pick_cent_p', y = 'trip_duration', data = data)
    plt.show()
    sns.regplot(x = 'bearing_drop_cent_p', y = 'trip_duration', data = data)
    plt.show()
    sns.regplot(x = 'bearing_cent_p_cent_d', y = 'trip_duration', data = data)
    plt.show()
    return

#label_pick,label_drop与duration的关系
def show_label():
    data = pd.read_csv('result.csv')
    label_pick = pd.DataFrame(data.groupby('label_pick')['trip_duration'].mean())
    label_pick.reset_index(inplace = True)
    label_drop = pd.DataFrame(data.groupby('label_drop')['trip_duration'].mean())
    label_drop.reset_index(inplace = True)
    
    label_pick['trip_duration'].plot(kind = 'line', rot = 0)
    plt.xlabel('label_pick')
    plt.ylabel('avg_trip_duration')
    plt.show()
    
    label_pick['trip_duration'].plot(kind = 'line', rot = 0)
    plt.xlabel('label_drop')
    plt.ylabel('avg_trip_duration')
    plt.show()

def show_centroid():
    data = pd.read_csv('result.csv')
    centroid_pick_long = pd.DataFrame(data.groupby('centroid_pick_long')['trip_duration'].mean())
    centroid_pick_long.reset_index(inplace = True)
    centroid_pick_lat = pd.DataFrame(data.groupby('centroid_pick_lat')['trip_duration'].mean())
    centroid_pick_lat.reset_index(inplace = True)
    centroid_drop_long = pd.DataFrame(data.groupby('centroid_drop_long')['trip_duration'].mean())
    centroid_drop_long.reset_index(inplace = True)
    centroid_drop_lat = pd.DataFrame(data.groupby('centroid_drop_lat')['trip_duration'].mean())
    centroid_drop_lat.reset_index(inplace = True)
    
    centroid_pick_long['trip_duration'].plot(kind = 'line', rot = 0)
    plt.xlabel('centroid_pick_long')
    plt.ylabel('avg_trip_duration')
    plt.show()
    
    centroid_pick_lat['trip_duration'].plot(kind = 'line', rot = 0)
    plt.xlabel('centroid_pick_lat')
    plt.ylabel('avg_trip_duration')
    plt.show()
    
    centroid_drop_long['trip_duration'].plot(kind = 'line', rot = 0)
    plt.xlabel('centroid_drop_long')
    plt.ylabel('avg_trip_duration')
    plt.show()
    
    centroid_drop_lat['trip_duration'].plot(kind = 'line', rot = 0)
    plt.xlabel('centroid_drop_lat')
    plt.ylabel('avg_trip_duration')
    plt.show()
    return

def show_hvsine_distance():
    train_cl = pd.read_csv('result.csv')
    
    hvsine_pick_cent_p = train_cl.loc[(train_cl.hvsine_pick_cent_p < 60)]
    sns.regplot(x = 'hvsine_pick_cent_p', y = 'trip_duration', data = hvsine_pick_cent_p)
    plt.show()
    hvsine_drop_cent_d = train_cl.loc[(train_cl.hvsine_drop_cent_d < 100)]
    sns.regplot(x = 'hvsine_drop_cent_d', y = 'trip_duration', data = hvsine_drop_cent_d)
    plt.show()
    hvsine_cent_p_cent_d = train_cl.loc[(train_cl.hvsine_cent_p_cent_d < 50)]
    sns.regplot(x = 'hvsine_cent_p_cent_d', y = 'trip_duration', data = hvsine_cent_p_cent_d)
    plt.show() 
    return

def show_manhtn_distance():
    train_cl = pd.read_csv('result.csv')
    
    manhtn_pick_cent_p = train_cl.loc[(train_cl.manhtn_pick_cent_p < 60)]
    sns.regplot(x = 'manhtn_pick_cent_p', y = 'trip_duration', data = manhtn_pick_cent_p)
    plt.show()
    manhtn_drop_cent_d = train_cl.loc[(train_cl.manhtn_drop_cent_d < 100)]
    sns.regplot(x = 'manhtn_drop_cent_d', y = 'trip_duration', data = manhtn_drop_cent_d)
    plt.show()
    manhtn_cent_p_cent_d = train_cl.loc[(train_cl.manhtn_cent_p_cent_d < 50)]
    sns.regplot(x = 'manhtn_cent_p_cent_d', y = 'trip_duration', data = manhtn_cent_p_cent_d)
    plt.show() 
    
    
plt.style.use({'figure.figsize':(12, 8)})
#show_manhtn_distance()
    