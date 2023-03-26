import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import copy
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from sklearn.linear_model import LinearRegression
import random
from pathlib import Path
import torch

data = pd.read_excel('modified_notebooks_data_frame_7_(132).xlsx')


plt.figure(figsize=(10,10))
cor = data.corr()
sns.heatmap(cor, annot=True, cmap='coolwarm')
plt.show()

X = data.drop('цена', axis=1)
y = data['цена']

train, val, test = np.split(data.sample(frac=1), [int(0.6*len(data)), int(0.8*len(data))])

def get_xy(dataframe, y_label, x_labels=None):
  dataframe = copy.deepcopy(dataframe)
  if x_labels is None:
    X = dataframe[[c for c in dataframe.columns if c!=y_label]].values
  else:
    if len(x_labels) == 1:
      X = dataframe[x_labels[0]].values.reshape(-1, 1)
    else:
      X = dataframe[x_labels].values

  y = dataframe[y_label].values.reshape(-1, 1)
  data = np.hstack((X, y))

  return data, X, y

_, X_train_all, y_train_all = get_xy(train, 'цена', x_labels=data.columns[:-1])
_, X_val_all, y_val_all = get_xy(val, 'цена', x_labels=data.columns[:-1])
_, X_test_all, y_test_all = get_xy(test, 'цена', x_labels=data.columns[:-1])

all_reg = LinearRegression()
all_reg.fit(X_train_all, y_train_all)

all_normalizer = tf.keras.layers.Normalization(input_shape=(13,), axis=-1)
all_normalizer.adapt(X_train_all)

print(data)

tf_lr_nn_model_0 = keras.models.load_model('npp_model_2')
tf_lr_nn_model_0.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=0.001), loss='mean_squared_error')

history = tf_lr_nn_model_0.fit(
    X_train_all, y_train_all,
    validation_data=(X_val_all, y_val_all),
    verbose=0, epochs=15000
)

def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.xlabel('Epoch')
  plt.ylabel('MSE')
  plt.legend()
  plt.grid(True)
  plt.show()

y_pred_lr = all_reg.predict(X_test_all)
y_pred_nn = tf_lr_nn_model_0.predict(X_test_all)

def MSE(y_pred, y_real):
  return (np.square(y_pred - y_real)).mean()

plot_loss(history)

MSE(y_pred_lr, y_test_all)

ax = plt.axes(aspect="equal")
plt.scatter(y_test_all, y_pred_lr, label="Lin Reg Preds")
plt.scatter(y_test_all, y_pred_nn, label="NN Preds")
plt.xlabel("True Values")
plt.ylabel("Predictions")
lims = [0, 300000]
plt.xlim(lims)
plt.ylim(lims)
plt.legend()
_ = plt.plot(lims, lims, c="red")

print(tf_lr_nn_model_0.predict([14.2,	3120,	2080,	4,	2,	8,	8,	16,	1024,	5,	12,	18,	4.7]))

print(tf_lr_nn_model_0.get_weights())

print(tf_lr_nn_model_0.get_metrics_result())

# Saving a trained model
tf_lr_nn_model_0.save('npp_model_4_1')
