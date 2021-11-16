from google.colab import drive
drive.mount('/content/drive')

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score, classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model

from tqdm.keras import TqdmCallback # Cool progress bar instead of the lame line-by-line "progress bar" that comes by default with Keras.

print(tf.__version__)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

# To suppress annoying warnings.
logging.getLogger('tensorflow').setLevel(logging.ERROR)
tf.autograph.set_verbosity(0) 