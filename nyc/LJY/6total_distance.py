import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.plot(train_df.trip_duration,train_df.total_distance)
plt.xlabel('trip_duration')
plt.ylabel('total_distance')
plt.show()
