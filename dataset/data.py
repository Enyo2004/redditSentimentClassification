# import the necessary libraries 
import pandas as pd 
import numpy as np

# get rid of tensorflow warnings
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['PYTHONUTF8'] = '1'

import tensorflow as tf 
import keras 


# import the dataset 
reddit = pd.read_csv('reddit_artist_posts_sentiment.csv')

# get the X and y variables 
X = reddit['text'] # independent variable 

y = reddit['label'] # dependent variable

# get the class names 
class_names = y.unique()


# convert the categorical data into numerical data
def classes(input):
    if input == class_names[0]: # if the input is negative
        return 0
    elif input == class_names[1]: # if the input is positive
        return 1
    else: # if the input is neutral
        return 2

# apply the function to y
y = y.apply(classes).astype(np.float32)

# split into training and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


# make the datasets for optimized data loading 
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(128).prefetch(tf.data.AUTOTUNE)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(128).prefetch(tf.data.AUTOTUNE)


