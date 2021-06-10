import numpy as np
import pandas as pd

#分割字符串来计算每个路径中的r，l，s的数量，step_dir形如"left|right"的字符串
#a_list是一个键值对数组，形如{(left,2)(right,3)}
#后面对a遍历得到每个step_dir中left，r，s的数量并返回。
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
        #print(a)
    if 'left' in (path.keys()):
        b = path['left']
        #print(b)
    if 'right' in (path.keys()):
        c = path['right']
        #print(c)
    return a,b,c

#直行，左转右转的数量，
def get_turn_number(dataframe):
    datafr1 = pd.read_csv("../data/fastest_routes_train_part_1.csv")
    datafr2 = pd.read_csv("../data/fastest_routes_train_part_2.csv")
    datafr = pd.concat([datafr1,datafr2])

    datafr['straight'] = 0
    datafr['right'] = 0
    datafr['left'] = 0
    datafr['straight'],datafr['left'],datafr['right'] = zip(*datafr['step_direction'].map(freq_turn))

    data_new = datafr[['id', 'total_distance', 'total_travel_time', 'straight','right','left']]
    dataframe = pd.merge(dataframe, data_new, on='id', how='left')
    return dataframe


#天气信息，最低温度和降水量,降雪量和降雪厚度
def get_weather_info(dataframe):
    weather = pd.read_csv("../data/weather_data.csv")

    #这里需要对于表格中的bool类型的数据处理成为浮点型变量
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
    dataframe = pd.merge(dataframe,weather[['date','minimum temperature',
                                            'precipitation','snow fall',
                                            'snow depth']],on = 'date',how = 'left')
    return dataframe

def get_holiday(dataframe):
    holiday = pd.read_csv('../data/holiday.csv')

    holiday['date'] = pd.to_datetime(holiday['date'])
    print(holiday.head())
    holiday['is_holiday'] = list(map(float, holiday['is_holiday']))
    dataframe['pickup_datetime'] = pd.to_datetime(dataframe['pickup_datetime'])
    dataframe['date'] = dataframe['pickup_datetime'].dt.date
    dataframe['date'] = pd.to_datetime(dataframe['date'])
    dataframe = pd.merge(dataframe,holiday[['date','is_holiday']],on='date',how='left')
    return dataframe