# Basic Libraries
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score, classification_report, confusion_matrix

# Tensorflow/Keras
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.utils import plot_model
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
print('TensorFlow Version: ', tf.__version__)

# Tensorboard
%load_ext tensorboard
# !rm -rf ./logs/ # Clears any logs from previous runs
log_dir = 'logs/fit/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
# %tensorboard --logdir logs/fit # To activate Tensorboard

# GDrive
from google.colab import drive
drive.mount('/content/drive')

# Misc.
from tqdm.keras import TqdmCallback # Cool progress bar instead of the lame line-by-line "progress bar" that comes by default with Keras.
import logging

# Warning supressors.
logging.getLogger('tensorflow').setLevel(logging.ERROR)
tf.autograph.set_verbosity(0) 
