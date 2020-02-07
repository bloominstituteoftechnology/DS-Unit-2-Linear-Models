# DS-Unit-Linear-Models 
EDible and poisonous mushrooms 
###Finding the differnce between edible mushrooms and posionous muchrooms. we will be using samples of 23 species of gilled mushrooms, of the Agaricus and Lepiota 



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


import pandas as pd

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data',header=None)

df.head()

df.columns = ['class', 'cap_shape', 'cap_surface', 'cap_color', 'bruises', 'odor', 'gill_attachment',\
                       'gill_spacing', 'gill_size', 'gill_color', 'stalk_shape', 'stalk_root',\
                       'stalk_surface_above_ring', 'stalk_surface_below_ring', 'stalk_color_above_ring',\
                       'stalk_color_below_ring', 'veil_type', 'veil_color', 'ring_number', 'ring_type',\
                       'spore_print_color', 'population', 'habitat']

df.head()

## as you can see there are two seperated columns of crosstab and histogram plots 

pd.crosstab(df['class'], df['population']).plot(kind = 'bar')


df['class'].hist()
