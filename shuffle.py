import pandas as pd
from sklearn.utils import shuffle
import seaborn as sns

data = pd.read_excel('modified_notebooks_data_frame_5_63.xlsx')

data = data.drop(['Unnamed: 0'], axis=1)

# data = shuffle(data)

# data.to_csv('modified_notebooks_data_frame_3_shuffled.csv')

data1 = pd.read_excel('modified_notebooks_data_frame_6.xlsx')
data1 = shuffle(data1)

# # data2 = pd.read_csv('modified_notebooks_data_frame_3_shuffled.csv')

data3 = pd.concat([data, data1], sort=False, axis=0)

data3.reset_index(drop=True , inplace=True)

# data3.drop(columns = ['Unnamed: 0'], axis = 1, inplace=True)

# print(data3)

data3.to_excel('modified_notebooks_data_frame_7_132.xlsx', index=False)
# sns.heatmap(data3.corr(), annot=True, cmap='coolwarm')

# sns.pairplot(data3.select_dtypes(['number'])) 
