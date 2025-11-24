# import the data of the dataframe
from dataset.data import *

# import libraries to see data
from matplotlib import pyplot as plt

# see the quantity of different labels in the dataset
label_count = reddit['label'].value_counts()
# print(label_count)

# plot the amount of different labels
reddit['label'].hist() 
#plt.show()


# print 10 random texts with its label
from random import randint 
sample = reddit.sample(n=10, random_state=randint(1, 1000))
# print(sample)

