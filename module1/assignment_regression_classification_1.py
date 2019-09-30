#%% [markdown]
# Lambda School Data Science, Unit 2: Predictive Modeling
# 
# # Regression & Classification, Module 1
# 
# ## Assignment
# 
# You'll use another **New York City** real estate dataset. 
# 
# But now you'll **predict how much it costs to rent an apartment**, instead of how much it costs to buy a condo.
# 
# The data comes from renthop.com, an apartment listing website.
# 
# - [ ] Look at the data. Choose a feature, and plot its relationship with the target.
# - [ ] Use scikit-learn for linear regression with one feature. You can follow the [5-step process from Jake VanderPlas](https://jakevdp.github.io/PythonDataScienceHandbook/05.02-introducing-scikit-learn.html#Basics-of-the-API).
# - [ ] Define a function to make new predictions and explain the model coefficient.
# - [ ] Organize and comment your code.
# 
# > [Do Not Copy-Paste.](https://docs.google.com/document/d/1ubOw9B3Hfip27hF2ZFnW3a3z9xAgrUDRReOEo-FHCVs/edit) You must type each of these exercises in, manually. If you copy and paste, you might as well not even do them. The point of these exercises is to train your hands, your brain, and your mind in how to read, write, and see code. If you copy-paste, you are cheating yourself out of the effectiveness of the lessons.
# 
# ## Stretch Goals
# - [ ] Do linear regression with two or more features.
# - [ ] Read [The Discovery of Statistical Regression](https://priceonomics.com/the-discovery-of-statistical-regression/)
# - [ ] Read [_An Introduction to Statistical Learning_](http://faculty.marshall.usc.edu/gareth-james/ISL/ISLR%20Seventh%20Printing.pdf), Chapter 2.1: What Is Statistical Learning?

#%%
import os, sys

get_ipython().system('ls')

#%%
# Ignore this Numpy warning when using Plotly Express:
# FutureWarning: Method .ptp is deprecated and will be removed in a future version. Use numpy.ptp instead.
# import warnings
# warnings.filterwarnings(action='ignore', category=FutureWarning, module='numpy')


#%%
# Read New York City apartment rental listing data
import pandas as pd
df = pd.read_csv('./data/apartments/renthop-nyc.csv')
assert df.shape == (49352, 34)


#%%
# Remove outliers: 
# the most extreme 1% prices,
# the most extreme .1% latitudes, &
# the most extreme .1% longitudes
df = df[(df['price'] >= 1375) & (df['price'] <= 15500) & 
        (df['latitude'] >=40.57) & (df['latitude'] < 40.99) &
        (df['longitude'] >= -74.1) & (df['longitude'] <= -73.38)]




#%%
df.describe()



#%%
df.isna().sum()

#%%
target = 'price'
features = df.columns[df.dtypes!='object']
features = features.drop(target)
df[features].isna().sum()

#%%
from sklearn.linear_model import LinearRegression
import numpy

lr_model = LinearRegression()

lr_model.fit(df[features],df[target])

#%%
import sklearn.metrics as metrics

y_pred = lr_model.predict(df[features])

mse = metrics.mean_squared_error(df[target], y_pred)
r2 = metrics.r2_score(df[target], y_pred)

mean = numpy.mean(df[target])
baseline = metrics.mean_squared_error(df[target],numpy.linspace(mean, mean, len(df[target])))

print(f'Baseline Mean Squared Error: {baseline}')

print(f'All numerics Mean Squared Error: {mse}')
print(f'All numerics R^2 score: {r2}')

# lr_model.coef_

# df.iloc[target]

#%%

def do_regression(df, features, target):
	lr_model = LinearRegression()
	lr_model.fit(df[features],df[target])
	y_pred = lr_model.predict(df[features])
	mse = metrics.mean_squared_error(df[target], y_pred)
	r2 = metrics.r2_score(df[target], y_pred)
	return(lr_model, mse, r2)

for feature in features:
	model, mse, r2 = do_regression(df, [feature], target)
	print(f'{target} changes by {model.coef_} as {feature} increases.')
	print(f'Mean Squared Error for this relationship: {mse}')
	print(f'R^2 score for this relationship: {r2}')


#%%
df.head()

#%%
import matplotlib.pyplot as pyplot
pyplot.rcParams['figure.facecolor'] = '#002B36'
pyplot.rcParams['axes.facecolor'] = 'black'


boxen = df.boxplot(	target, 
					by='bathrooms', 
					patch_artist=True,
					return_type='dict',
					figsize=(8,10))['price']

for linetype in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
	pyplot.setp(boxen[linetype], color='crimson', linewidth=4)

pyplot.setp(boxen['boxes'], facecolor='white')
pyplot.setp(boxen['fliers'], markeredgecolor='crimson')
pyplot.grid(color='#666666')
pyplot.ylabel('price')
pyplot.title('Price of apartments by bathroom count')
pyplot.suptitle(None)

pyplot.show()



#%%
