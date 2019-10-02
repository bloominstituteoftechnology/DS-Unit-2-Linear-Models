#%% [markdown]
# Lambda School Data Science, Unit 2: Predictive Modeling
# 
# # Regression & Classification, Module 3
# 
# ## Assignment
# 
# We're going back to our other **New York City** real estate dataset. Instead of predicting apartment rents, you'll predict property sales prices.
# 
# But not just for condos in Tribeca...
# 
# Instead, predict property sales prices for **One Family Dwellings** (`BUILDING_CLASS_CATEGORY` == `'01 ONE FAMILY DWELLINGS'`). 
# 
# Use a subset of the data where the **sale price was more than \\$100 thousand and less than $2 million.** 
# 
# The [NYC Department of Finance](https://www1.nyc.gov/site/finance/taxes/property-rolling-sales-data.page) has a glossary of property sales terms and NYC Building Class Code Descriptions. The data comes from the [NYC OpenData](https://data.cityofnewyork.us/browse?q=NYC%20calendar%20sales) portal.
# 
# - [ ] Do train/test split. Use data from January — March 2019 to train. Use data from April 2019 to test.
# - [ ] Do one-hot encoding of categorical features.
# - [ ] Do feature selection with `SelectKBest`.
# - [ ] Do [feature scaling](https://scikit-learn.org/stable/modules/preprocessing.html).
# - [ ] Fit a ridge regression model with multiple features.
# - [ ] Get mean absolute error for the test set.
# - [ ] As always, commit your notebook to your fork of the GitHub repo.
# 
# 
# ## Stretch Goals
# - [ ] Add your own stretch goal(s) !
# - [ ] Instead of `RidgeRegression`, try `LinearRegression`. Depending on how many features you select, your errors will probably blow up! 💥
# - [ ] Instead of `RidgeRegression`, try [`RidgeCV`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html).
# - [ ] Learn more about feature selection:
#	 - ["Permutation importance"](https://www.kaggle.com/dansbecker/permutation-importance)
#	 - [scikit-learn's User Guide for Feature Selection](https://scikit-learn.org/stable/modules/feature_selection.html)
#	 - [mlxtend](http://rasbt.github.io/mlxtend/) library
#	 - scikit-learn-contrib libraries: [boruta_py](https://github.com/scikit-learn-contrib/boruta_py) & [stability-selection](https://github.com/scikit-learn-contrib/stability-selection)
#	 - [_Feature Engineering and Selection_](http://www.feat.engineering/) by Kuhn & Johnson.
# - [ ] Try [statsmodels](https://www.statsmodels.org/stable/index.html) if you’re interested in more inferential statistical approach to linear regression and feature selection, looking at p values and 95% confidence intervals for the coefficients.
# - [ ] Read [_An Introduction to Statistical Learning_](http://faculty.marshall.usc.edu/gareth-james/ISL/ISLR%20Seventh%20Printing.pdf), Chapters 1-3, for more math & theory, but in an accessible, readable way.
# - [ ] Try [scikit-learn pipelines](https://scikit-learn.org/stable/modules/compose.html).

#%%


#%%
import pandas as pd
# import pandas_profiling

# Read New York City property sales data
df = pd.read_csv('./data/condos/NYC_Citywide_Rolling_Calendar_Sales.csv')

# Change column names: replace spaces with underscores
df.columns = [col.replace(' ', '_') for col in df]

# SALE_PRICE was read as strings.
# Remove symbols, convert to integer
df['SALE_PRICE'] = (
	df['SALE_PRICE']
	.str.replace('$','')
	.str.replace('-','')
	.str.replace(',','')
	.astype(int)
)


#%%
# BOROUGH is a numeric column, but arguably should be a categorical feature,
# so convert it from a number to a string
df['BOROUGH'] = df['BOROUGH'].astype(str)


#%%
# Reduce cardinality for NEIGHBORHOOD feature

# Get a list of the top 10 neighborhoods
top10 = df['NEIGHBORHOOD'].value_counts()[:10].index

# At locations where the neighborhood is NOT in the top 10, 
# replace the neighborhood with 'OTHER'
df.loc[~df['NEIGHBORHOOD'].isin(top10), 'NEIGHBORHOOD'] = 'OTHER'

#%%
import pandas
print(df.shape)
df.isna().sum()
# df.loc[7]

#%%

df_ofd = df[(df['BUILDING_CLASS_CATEGORY'] == '01 ONE FAMILY DWELLINGS') &
			(df['SALE_PRICE'] > 100000) &
			(df['SALE_PRICE'] < 2000000)]
df_ofd['sale_dt'] = pandas.to_datetime(df_ofd['SALE_DATE'])

#%%
print(df_ofd.isna().sum())

print(df_ofd.shape)
df_ofd = df_ofd.dropna(subset=['ZIP_CODE'])
print(df_ofd.shape)
df_ofd = df_ofd.dropna(axis=1)
print(df_ofd.shape)

df_ofd.isna().sum()

#%%
df_ofd.head()


#%%
import category_encoders
from typing import Optional
import numpy

def clean(	frame:pandas.DataFrame, 
			cols:Optional[list] = None,
			exclude_cols:Optional[list] = None,
			max_cardinality:Optional[int] = None) -> pandas.DataFrame:
	"""
	Cleans and one-hot encodes the dataframe.
	
	Args:
		frame (pandas.DataFrame): Dataframe to clean
		cols (list, optional): Columns to one-hot encode. Defaults to all string columns.
		exclude_cols (list, optional): Columns to skip one-hot encoding. Defaults to None.
		max_cardinality (int, optional): Maximum cardinality of columns to encode. Defaults to no maximum cardinality.
	
	Returns:
		pandas.DataFrame: The cleaned dataframe.
	"""


	cleaned = frame.copy()

	if cols is None: cols = list(cleaned.columns[cleaned.dtypes=='object'])

	if exclude_cols is not None:
		for col in exclude_cols:
			cols.remove(col)

	if max_cardinality is not None:
		described = cleaned[cols].describe(exclude=[numpy.number])
		cols = list(described.columns[described.loc['unique'] <= max_cardinality])

	encoder = category_encoders.OneHotEncoder(return_df=True, use_cat_names=True, cols=cols)
	cleaned = encoder.fit_transform(cleaned)

	return(cleaned)

#%%

cleaned = clean(df_ofd, max_cardinality=100, exclude_cols=['SALE_DATE'])

#%%
list(cleaned.columns)

#%%

target = 'SALE_PRICE'
features = cleaned.columns[cleaned.dtypes!='object'].drop([target, 'sale_dt'])

#%%
list(features)

#%%
len(features)

#%%

train = cleaned[(df_ofd['sale_dt'] < pandas.to_datetime('2019-04-01')) &
				(df_ofd['sale_dt'] >= pandas.to_datetime('2019-01-01'))]

test = cleaned[	(df_ofd['sale_dt'] < pandas.to_datetime('2019-05-01')) &
				(df_ofd['sale_dt'] >= pandas.to_datetime('2019-04-01'))]

#%%

#%%

#%%
train.isna().sum()[train.isna().sum() > 0]

#%%
test.shape

#%%
from sklearn.feature_selection import f_regression, SelectKBest
from sklearn.linear_model import LinearRegression
import sklearn.metrics as metrics

baseline = numpy.linspace(test[target].mean(), test[target].mean(), num=len(test[target]))

r2 = metrics.r2_score(test[target], baseline)
rmse = numpy.sqrt(metrics.mean_squared_error(test[target], baseline))
mae = metrics.mean_absolute_error(test[target], baseline)
print(f'r-squared score for the baseline: {r2}')
print(f'RMSE for the baseline: {rmse}')
print(f'MAE for the baseline: {mae}')

#%%

r2s_linear = []
for i in range(1,train[features].shape[1]+1):
	selector = SelectKBest(score_func=f_regression, k=i)
	train_selected = selector.fit_transform(train[features], train[target])
	test_selected = selector.transform(test[features])
	# print(features[selector.get_support()])
	lr_model = LinearRegression()
	lr_model.fit(train_selected, train[target])
	predicted = lr_model.predict(test_selected)
	r2 = metrics.r2_score(test[target], predicted)
	rmse = numpy.sqrt(metrics.mean_squared_error(test[target], predicted))
	mae = metrics.mean_absolute_error(test[target], predicted)
	# print(f'r-squared score for the {i} best features: {r2}')
	# print(f'RMSE for the {i} best features: {rmse}')
	# print(f'MAE for the {i} best features: {mae}')
	r2s_linear.append(r2)

#%%

from sklearn.linear_model import Ridge

r2s_ridge = []
alphas = [(i/20)**2 for i in range(1,203, 2)]
for a in alphas:
	ridge_model = Ridge(alpha=a)
	ridge_model.fit(train[features], train[target])
	predicted = ridge_model.predict(test[features])

	r2 = metrics.r2_score(test[target], predicted)
	rmse = numpy.sqrt(metrics.mean_squared_error(test[target], predicted))
	# print(f'r-squared score at alpha={(i/10)**2}: {r2}')
	# print(f'RMSE at alpha={(i/10)**2}: {rmse}')
	r2s_ridge.append(r2)

#%%
import matplotlib.pyplot as pyplot
pyplot.rcParams['figure.facecolor'] = '#002B36'
pyplot.rcParams['axes.facecolor'] = 'black'
pyplot.rcParams['figure.figsize'] = (10,8)

pyplot.plot(r2s_linear)
pyplot.title('Linear Regression')
pyplot.xlabel('Number of features (KBest)')
pyplot.ylabel('R^2 score')
pyplot.ylim(0, .45)
pyplot.show()

pyplot.plot(alphas, r2s_ridge)
pyplot.title('Ridge Regression')
pyplot.xlabel('Alpha')
pyplot.xscale('log')
pyplot.ylabel('R^2 score')
pyplot.ylim(0, .45)
pyplot.show()



#%%


#%%
