import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import torch
import joblib
# from PT_multiple_linear_regression_nn import MultipleLinearRegressionModel

# npp_model1 = keras.models.load_model('notebooks_price_predict_model_1.h5')

# npp_model2 = keras.models.load_model('notebooks_price_predict_model_1')

# npp_model3 = keras.models.load_model('npp_model_1(1)')

npp_model4 = keras.models.load_model('npp_model_4_1')

npp_model5 = torch.jit.load('model5_1_scripted.pt')
npp_model5.eval()

# npp_model6 = keras.models.load_model('npp_model_6')

npp_model7 = joblib.load('random_forest_model.joblib')

data = pd.read_excel('modified_notebooks_data_frame_7_132.xlsx')
# data = data.drop(['Unnamed: 0'], axis=1)

sort_data = data.sort_values(by='цена', ascending=True)

sort_data.reset_index(drop=True , inplace=True)

# # converting DataFrame to NumPy array
notebooks = sort_data.to_numpy()

# # converting NumPy array into PyTorch tensor
notebooks1 = torch.from_numpy(notebooks)

# # Testing models through NumPy array and PyTorch tensor
for notebook in notebooks:
    tf.cast(notebook, dtype=float)
    print(f'Model 4 predict: {npp_model4(notebook[:-1])}, Accuracy: {round(np.squeeze(npp_model4(notebook[:-1]))[()]/notebook[-1] * 100, 2)}%, True price: {notebook[-1]}')   
for notebook in notebooks:
    print(f'Model 7 predict: {npp_model7.predict([notebook[:-1]])}, Accuracy: {npp_model7.predict([notebook[:-1]])/notebook[-1] * 100}%, True price: {notebook[-1]}')    
for notebook in notebooks1:
    notebook = notebook.type(torch.FloatTensor)
    print(f'Model 5 predict: {npp_model5(notebook[:-1])}, Accuracy: {npp_model5(notebook[:-1]).squeeze()/notebook[-1] * 100}%, True price: {notebook[-1]}')

t1 = np.array([21.5,	1920,	1080,	3,	1,	3,	7,	4,	256,	4,	2,	4,	3.2])
t2 = np.array([15.6,	1920,	1080,	3,	2,	8,	15,	32,	2048,	3,	6,	16,	4.4])
t3 = np.array([15.6,	1366,	768,	1,	2,	1,	6, 4,	128,	4,	2,	4,	2.8])
t4 = np.array([15.6,	1920,	1080,	3,	1,	4,	11, 8,	256,	5,	2,	4,	4.1])
print('=' * 50)
print(npp_model4(tf.cast(t1, dtype=float)), npp_model5(torch.from_numpy(t1).type(torch.FloatTensor)), npp_model7.predict([[21.5,	1920,	1080,	3,	1,	3,	7,	4,	256,	4,	2,	4,	3.2]]))
print(npp_model4(tf.cast(t2, dtype=float)), npp_model5(torch.from_numpy(t2).type(torch.FloatTensor)), npp_model7.predict([[15.6,	1920,	1080,	3,	2,	8,	15,	32,	2048,	3,	6,	16,	4.4]]))
print(npp_model4(tf.cast(t3, dtype=float)), npp_model5(torch.from_numpy(t3).type(torch.FloatTensor)), npp_model7.predict([[15.6,	1366,	768,	1,	2,	1,	6, 4,	128,	4,	2,	4,	2.8]]))
print(npp_model4(tf.cast(t4, dtype=float)), npp_model5(torch.from_numpy(t4).type(torch.FloatTensor)), npp_model7.predict([[15.6,	1920,	1080,	3,	1,	4,	11, 8,	256,	5,	2,	4,	4.1]]))