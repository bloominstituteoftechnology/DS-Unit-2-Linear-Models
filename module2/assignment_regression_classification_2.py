#%% [markdown]
# Lambda School Data Science, Unit 2: Predictive Modeling
# 
# # Regression & Classification, Module 2
# 
# ## Assignment
# 
# You'll continue to **predict how much it costs to rent an apartment in NYC,** using the dataset from renthop.com.
# 
# - [ ] Do train/test split. Use data from April & May 2016 to train. Use data from June 2016 to test.
# - [ ] Engineer at least two new features. (See below for explanation & ideas.)
# - [ ] Fit a linear regression model with at least two features.
# - [ ] Get the model's coefficients and intercept.
# - [ ] Get regression metrics RMSE, MAE, and $R^2$, for both the train and test data.
# - [ ] What's the best test MAE you can get? Share your score and features used with your cohort on Slack!
# - [ ] As always, commit your notebook to your fork of the GitHub repo.
# 
# 
# #### [Feature Engineering](https://en.wikipedia.org/wiki/Feature_engineering)
# 
# > "Some machine learning projects succeed and some fail. What makes the difference? Easily the most important factor is the features used." — Pedro Domingos, ["A Few Useful Things to Know about Machine Learning"](https://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf)
# 
# > "Coming up with features is difficult, time-consuming, requires expert knowledge. 'Applied machine learning' is basically feature engineering." — Andrew Ng, [Machine Learning and AI via Brain simulations](https://forum.stanford.edu/events/2011/2011slides/plenary/2011plenaryNg.pdf) 
# 
# > Feature engineering is the process of using domain knowledge of the data to create features that make machine learning algorithms work. 
# 
# #### Feature Ideas
# - Does the apartment have a description?
# - How long is the description?
# - How many total perks does each apartment have?
# - Are cats _or_ dogs allowed?
# - Are cats _and_ dogs allowed?
# - Total number of rooms (beds + baths)
# - Ratio of beds to baths
# - What's the neighborhood, based on address or latitude & longitude?
# 
# ## Stretch Goals
# - [ ] If you want more math, skim [_An Introduction to Statistical Learning_](http://faculty.marshall.usc.edu/gareth-james/ISL/ISLR%20Seventh%20Printing.pdf),  Chapter 3.1, Simple Linear Regression, & Chapter 3.2, Multiple Linear Regression
# - [ ] If you want more introduction, watch [Brandon Foltz, Statistics 101: Simple Linear Regression](https://www.youtube.com/watch?v=ZkjP5RJLQF4)
# (20 minutes, over 1 million views)
# - [ ] Add your own stretch goal(s) !

#%%
import os, sys

#%%
# Ignore this Numpy warning when using Plotly Express:
# FutureWarning: Method .ptp is deprecated and will be removed in a future version. Use numpy.ptp instead.
# import warnings
# warnings.filterwarnings(action='ignore', category=FutureWarning, module='numpy')


#%%
import numpy
import pandas

#%%

# Read New York City apartment rental listing data
df = pandas.read_csv('./data/apartments/renthop-nyc.csv')
assert df.shape == (49352, 34)

# Remove the most extreme 1% prices,
# the most extreme .1% latitudes, &
# the most extreme .1% longitudes
df = df[(df['price'] >= numpy.percentile(df['price'], 0.5)) & 
        (df['price'] <= numpy.percentile(df['price'], 99.5)) & 
        (df['latitude'] >= numpy.percentile(df['latitude'], 0.05)) & 
        (df['latitude'] < numpy.percentile(df['latitude'], 99.95)) &
        (df['longitude'] >= numpy.percentile(df['longitude'], 0.05)) & 
        (df['longitude'] <= numpy.percentile(df['longitude'], 99.95))]

#%%
df.head()

#%%
df['display_address'].value_counts()[df['display_address'].value_counts() <= 5].index

#%%
cleaned = df.copy()
cleaned['has_description'] = ((cleaned['description'].isna()==False) & (cleaned['description'].str.strip().str.len() > 0)).replace({False: 0, True: 1})
cleaned['created_dt'] = pandas.to_datetime(cleaned['created'])

cleaned['created_week'] = cleaned['created_dt'].dt.weekofyear
cleaned['interest_numeric'] = cleaned['interest_level'].replace({'low':1,'medium':2,'high':3})
# cleaned['is_broadway'] = cleaned['display_address']=='Broadway'
cleaned['display_address'] = cleaned['display_address'].str.strip().str.lower()
top_addresses = list(cleaned['display_address'].value_counts()[cleaned['display_address'].value_counts() >= 10].index)
cleaned['top_addresses'] = cleaned['display_address'].where(cleaned['display_address'].isin(top_addresses), other='other')
cleaned = cleaned.join(pandas.get_dummies(cleaned['top_addresses'], prefix='address_'))

# cleaned['perk_count'] = cleaned[['elevator', 'cats_allowed', 'hardwood_floors', 'dogs_allowed', 'doorman', 'dishwasher', 'no_fee', 'laundry_in_building', 'fitness_center', 'pre-war', 'laundry_in_unit', 'roof_deck', 'outdoor_space', 'dining_room', 'high_speed_internet', 'balcony', 'swimming_pool', 'new_construction', 'terrace', 'exclusive', 'loft', 'garden_patio', 'wheelchair_access', 'common_outdoor_space']].sum(axis=1)
#%%
# top_addresses
cleaned['top_addresses'].value_counts()

#%%
# cleaned


#%%
import sklearn.model_selection as model_selection

train, test = model_selection.train_test_split(cleaned)


#%%
cleaned.columns


#%%
# test.loc[11956]

#%%
target = 'price'
features = cleaned.columns[cleaned.dtypes!='object']
features = features.drop(target)
features = features.drop('created_dt')
# features = ['bathrooms', 'bedrooms', 'longitude', 'elevator', 'doorman', 'dishwasher', 'fitness_center', 'laundry_in_unit', 'dining_room', 'interest_numeric']

# ['bathrooms', 'bedrooms', 'interest_numeric', 'longitude', 'elevator', 'doorman', 'terrace', 'dishwasher', 'fitness_center']
# cleaned[features].isna().sum()

#%%
from sklearn.linear_model import LinearRegression
import numpy

lr_model = LinearRegression(normalize=True)

lr_model.fit(train[features],train[target])

#%%
import sklearn.metrics as metrics

y_train = lr_model.predict(train[features])
y_test = lr_model.predict(test[features])

train_rmse = numpy.sqrt(metrics.mean_squared_error(train[target], y_train))
train_mae = metrics.mean_absolute_error(train[target], y_train)
train_r2 = metrics.r2_score(train[target], y_train)
test_rmse = numpy.sqrt(metrics.mean_squared_error(test[target], y_test))
test_mae = metrics.mean_absolute_error(test[target], y_test)
test_r2 = metrics.r2_score(test[target], y_test)

mean = numpy.mean(cleaned[target])
baseline_rmse = numpy.sqrt(metrics.mean_squared_error(cleaned[target],numpy.linspace(mean, mean, len(cleaned[target]))))
baseline_mae = metrics.mean_absolute_error(cleaned[target], numpy.linspace(mean, mean, len(cleaned[target])))
baseline_r2 = metrics.r2_score(cleaned[target],numpy.linspace(mean, mean, len(cleaned[target])))

print(f'Features: {features}')
print(f'Baseline Root Mean Squared Error: {baseline_rmse}')
print(f'Baseline Mean Absolute Error: {baseline_mae}')
print(f'Baseline R^2 score: {baseline_r2}')
print(f'Train Root Mean Squared Error: {train_rmse}')
print(f'Train Mean Absolute Error: {train_mae}')
print(f'Train R^2 score: {train_r2}')
print(f'Test Root Mean Squared Error: {test_rmse}')
print(f'Test Mean Absolute Error: {test_mae}')
print(f'Test R^2 score: {test_r2}')




#%%

# cleaned.corr().loc['price']


#%%
