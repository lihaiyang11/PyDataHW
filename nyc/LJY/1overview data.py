import pandas as pd  #pandas for using dataframe and reading csv
train_fr_1 = pd.read_csv('../data/fastest_routes_train_part_1.csv')
train_fr_2 = pd.read_csv('../data/fastest_routes_train_part_2.csv')
train_fr = pd.concat([train_fr_1, train_fr_2])#合并
train_fr_new = train_fr[['id', 'total_distance', 'total_travel_time', 'number_of_steps']] #提取形成新的表
train_df = pd.read_csv('../data/train.csv')
train = pd.merge(train_df, train_fr_new, on = 'id', how = 'left') #根据traindf找new
train_df = train.copy()
train_df.info() #列出列名，数量，数据类型
print(train_df.columns.values) #直接导出列名
