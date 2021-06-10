import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.plot(train_df.trip_duration,train_df.number_of_steps)
plt.xlabel('trip_duration')
plt.ylabel('number_of_steps')
plt.show()

