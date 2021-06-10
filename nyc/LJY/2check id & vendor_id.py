train_data = train_df.copy()
print("Number of columns and rows and columns are {} and {} respectively.".format(train_data.shape[1], train_data.shape[0]))
if train_data.id.nunique() == train_data.shape[0]:
    print("Train ids are unique")
print("Number of Nulls - {}.".format(train_data.isnull().sum().sum()))
#Pandas会用NaN（not a number）来表示一个缺失的数据,有一个函数isnull()可以直接判断该列中的哪个数据为NaN,检查缺省值

plt.plot(train_df.trip_duration,train_df.vendor_id)
plt.xlabel('trip_duration')
plt.ylabel('vendor_id')
