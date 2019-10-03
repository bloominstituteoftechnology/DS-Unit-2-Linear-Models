#%% [markdown]
# Lambda School Data Science, Unit 2: Predictive Modeling
# 
# # Regression & Classification, Module 4
# 
# 
# ## Assignment
# 
# - [ ] Watch Aaron's [video #1](https://www.youtube.com/watch?v=pREaWFli-5I) (12 minutes) & [video #2](https://www.youtube.com/watch?v=bDQgVt4hFgY) (9 minutes) to learn about the mathematics of Logistic Regression.
# - [ ] [Sign up for a Kaggle account](https://www.kaggle.com/), if you don’t already have one. Go to our Kaggle InClass competition website. You will be given the URL in Slack. Go to the Rules page. Accept the rules of the competition.
# - [ ] Do train/validate/test split with the Tanzania Waterpumps data.
# - [ ] Begin with baselines for classification.
# - [ ] Use scikit-learn for logistic regression.
# - [ ] Get your validation accuracy score.
# - [ ] Submit your predictions to our Kaggle competition. (Go to our Kaggle InClass competition webpage. Use the blue **Submit Predictions** button to upload your CSV file. Or you can use the Kaggle API to submit your predictions.)
# - [ ] Commit your notebook to your fork of the GitHub repo.
# 
# ---
# 
# 
# ## Stretch Goals
# 
# - [ ] Add your own stretch goal(s) !
# - [ ] Clean the data. For ideas, refer to [The Quartz guide to bad data](https://github.com/Quartz/bad-data-guide),  a "reference to problems seen in real-world data along with suggestions on how to resolve them." One of the issues is ["Zeros replace missing values."](https://github.com/Quartz/bad-data-guide#zeros-replace-missing-values)
# - [ ] Make exploratory visualizations.
# - [ ] Do one-hot encoding. For example, you could try `quantity`, `basin`, `extraction_type_class`, and more. (But remember it may not work with high cardinality categoricals.)
# - [ ] Do [feature scaling](https://scikit-learn.org/stable/modules/preprocessing.html).
# - [ ] Get and plot your coefficients.
# - [ ] Try [scikit-learn pipelines](https://scikit-learn.org/stable/modules/compose.html).
# 
# ---
# 
# ## Data Dictionary 
# 
# ### Features
# 
# Your goal is to predict the operating condition of a waterpoint for each record in the dataset. You are provided the following set of information about the waterpoints:
# 
# - `amount_tsh` : Total static head (amount water available to waterpoint)
# - `date_recorded` : The date the row was entered
# - `funder` : Who funded the well
# - `gps_height` : Altitude of the well
# - `installer` : Organization that installed the well
# - `longitude` : GPS coordinate
# - `latitude` : GPS coordinate
# - `wpt_name` : Name of the waterpoint if there is one
# - `num_private` :  
# - `basin` : Geographic water basin
# - `subvillage` : Geographic location
# - `region` : Geographic location
# - `region_code` : Geographic location (coded)
# - `district_code` : Geographic location (coded)
# - `lga` : Geographic location
# - `ward` : Geographic location
# - `population` : Population around the well
# - `public_meeting` : True/False
# - `recorded_by` : Group entering this row of data
# - `scheme_management` : Who operates the waterpoint
# - `scheme_name` : Who operates the waterpoint
# - `permit` : If the waterpoint is permitted
# - `construction_year` : Year the waterpoint was constructed
# - `extraction_type` : The kind of extraction the waterpoint uses
# - `extraction_type_group` : The kind of extraction the waterpoint uses
# - `extraction_type_class` : The kind of extraction the waterpoint uses
# - `management` : How the waterpoint is managed
# - `management_group` : How the waterpoint is managed
# - `payment` : What the water costs
# - `payment_type` : What the water costs
# - `water_quality` : The quality of the water
# - `quality_group` : The quality of the water
# - `quantity` : The quantity of water
# - `quantity_group` : The quantity of water
# - `source` : The source of the water
# - `source_type` : The source of the water
# - `source_class` : The source of the water
# - `waterpoint_type` : The kind of waterpoint
# - `waterpoint_type_group` : The kind of waterpoint
# 
# ### Labels
# 
# There are three possible values:
# 
# - `functional` : the waterpoint is operational and there are no repairs needed
# - `functional needs repair` : the waterpoint is operational, but needs repairs
# - `non functional` : the waterpoint is not operational
# 
# --- 
# 
# ## Generate a submission
# 
# Your code to generate a submission file may look like this:
# 
# ```python
# # estimator is your model or pipeline, which you've fit on X_train
# 
# # X_test is your pandas dataframe or numpy array, 
# # with the same number of rows, in the same order, as test_features.csv, 
# # and the same number of columns, in the same order, as X_train
# 
# y_pred = estimator.predict(X_test)
# 
# 
# # Makes a dataframe with two columns, id and status_group, 
# # and writes to a csv file, without the index
# 
# sample_submission = pandas.read_csv('sample_submission.csv')
# submission = sample_submission.copy()
# submission['status_group'] = y_pred
# submission.to_csv('your-submission-filename.csv', index=False)
# ```
# 
# If you're working locally, the csv file is saved in the same directory as your notebook.
# 
# If you're using Google Colab, you can use this code to download your submission csv file.
# 
# ```python
# from google.colab import files
# files.download('your-submission-filename.csv')
# ```
# 
# ---

#%%

#%%
# Read the Tanzania Waterpumps data
# train_features.csv : the training set features
# train_labels.csv : the training set labels
# test_features.csv : the test set features
# sample_submission.csv : a sample submission file in the correct format
    
import pandas
import numpy

train_features = pandas.read_csv('./data/waterpumps/train_features.csv')
train_labels = pandas.read_csv('./data/waterpumps/train_labels.csv')
test_features = pandas.read_csv('./data/waterpumps/test_features.csv')
sample_submission = pandas.read_csv('./data/waterpumps/sample_submission.csv')

assert train_features.shape == (59400, 40)
assert train_labels.shape == (59400, 2)
assert test_features.shape == (14358, 40)
assert sample_submission.shape == (14358, 2)


#%%
train_features.head()

#%%
train_features.isna().sum()
# We end up disregarding these -
# As they're all categorical variables, one-hot encoding
# takes care of our NaNs for us

#%%
train_labels.head()

#%%
train_kaggle = pandas.merge(train_features, train_labels, on='id')
train_kaggle.head()

#%%
train_features.describe(exclude=[numpy.number])

#%% [markdown]
#
# ## One-hot encoding and pre-split cleaning
#

#%%
import category_encoders
from typing import Optional
import numpy

def keepTopN(	column:pandas.Series,
				n:int,
				default:Optional[object] = None) -> pandas.Series:
	"""
	Keeps the top n most popular values of a Series, while replacing the rest with `default`
	
	Args:
		column (pandas.Series): Series to operate on
		n (int): How many values to keep
		default (object, optional): Defaults to NaN. Value with which to replace remaining values
	
	Returns:
		pandas.Series: Series with the most popular n values
	"""

	if default is None: default = numpy.nan

	val_counts = column.value_counts()
	if n > len(val_counts): n = len(val_counts)
	top_n = list(val_counts[:n].index)
	return(column.where(column.isin(top_n), other=default))

def oneHot(	frame:pandas.DataFrame, 
			cols:Optional[list] = None,
			exclude_cols:Optional[list] = None,
			max_cardinality:Optional[int] = None) -> pandas.DataFrame:
	"""
	One-hot encodes the dataframe.
	
	Args:
		frame (pandas.DataFrame): Dataframe to clean
		cols (list, optional): Columns to one-hot encode. Defaults to all string columns.
		exclude_cols (list, optional): Columns to skip one-hot encoding. Defaults to None.
		max_cardinality (int, optional): Maximum cardinality of columns to encode. Defaults to no maximum cardinality.
	
	Returns:
		pandas.DataFrame: The one_hot_encoded dataframe.
	"""


	one_hot_encoded = frame.copy()

	if cols is None: cols = list(one_hot_encoded.columns[one_hot_encoded.dtypes=='object'])

	if exclude_cols is not None:
		for col in exclude_cols:
			cols.remove(col)

	if max_cardinality is not None:
		described = one_hot_encoded[cols].describe(exclude=[numpy.number])
		cols = list(described.columns[described.loc['unique'] <= max_cardinality])

	encoder = category_encoders.OneHotEncoder(return_df=True, use_cat_names=True, cols=cols)
	one_hot_encoded = encoder.fit_transform(one_hot_encoded)

	return(one_hot_encoded)

#%%
def clean_X(df, max_ordinality=100, int_ts=False):

	cleaned = df.copy().drop(columns=['recorded_by'])

	categorical_description = cleaned.describe(exclude=[numpy.number])
	if int_ts: 
		cat_cols = categorical_description.drop(columns=['date_recorded']).columns
	else:
		cat_cols = categorical_description.columns
	# high_ordinality_cols = categorical_description[categorical_description.loc['unique'] > max_ordinality].columns
	
	for col in cat_cols:
		cleaned[col] = keepTopN(cleaned[col], max_ordinality, default='other')

	if int_ts:
		cleaned['date_recorded_dt'] = pandas.to_datetime(df['date_recorded'])
		cleaned['date_recorded_ts'] = cleaned['date_recorded_dt'].view('int64')

		return(cleaned.drop(columns=['date_recorded_dt', 'date_recorded']))
	else:
		return(cleaned)

#%%
train_targets = train_labels.sort_values(by=['id'])['status_group'].replace({'functional': 1, 'functional needs repair': 2, 'non functional': 3})

#%%
train_targets.isna().sum()

#%%
train_targets

#%%

combined = pandas.concat([train_features, test_features])

cleaned_combined = oneHot(clean_X(combined, max_ordinality=200, int_ts=True))#100))
cleaned_train = cleaned_combined[cleaned_combined['id'].isin(train_features['id'])].sort_values(by=['id'])
cleaned_test = cleaned_combined[cleaned_combined['id'].isin(test_features['id'])].sort_values(by=['id'])
assert list(cleaned_train.columns) == list(cleaned_test.columns)
assert list(cleaned_train['id']) == list(train_labels.sort_values(by=['id'])['id'])

#%%
set(cleaned_train.columns) - set(cleaned_test.columns)

#%%
pandas.set_option('display.max_columns', 500)
# train_features.describe(exclude=[numpy.number])
#%%
dict(cleaned_combined.dtypes)

#%%
cleaned_test

#%%
import sklearn.preprocessing as preprocessing

scaler = preprocessing.StandardScaler()
scaled_train = scaler.fit_transform(cleaned_train.drop(columns=['id']))
scaled_train_ids = cleaned_train['id']
scaled_test = scaler.transform(cleaned_test.drop(columns=['id']))
scaled_test_ids = cleaned_test['id']

#%%

#%%
import sklearn.linear_model as linear_model
import sklearn.model_selection as model_selection

X_train_train, X_train_test, y_train_train, y_train_test = model_selection.train_test_split(scaled_train, train_targets, random_state=1)

print(f'X_train_train: {X_train_train.shape}')
print(f'X_train_test: {X_train_test.shape}')
print(f'y_train_train: {y_train_train.shape}')
print(f'y_train_test: {y_train_test.shape}')

lr_model = linear_model.LogisticRegression(solver='lbfgs', multi_class='auto')
lr_model.fit(X_train_train, y_train_train)

#%%
y_pred_train_train = lr_model.predict(X_train_train)
y_pred_train_test = lr_model.predict(X_train_test)
y_pred_train_test.shape

#%%
y_train_test.shape

#%%
y_pred_train_test

#%%
from scipy.stats import mode
mode(train_targets)

#%%
import sklearn.metrics as metrics
print(f'Baseline (all functional) for train.train: {metrics.accuracy_score(y_train_train, numpy.linspace(1,1,y_train_train.shape[0]))}')

#%%
print('Logistic Regression score:')
lr_model.score(X_train_train,y_train_train)

#%%
metrics.accuracy_score(y_train_train, y_pred_train_train)

#%%
print(f'Baseline (all functional) for train.test: {metrics.accuracy_score(y_train_test, numpy.linspace(1,1,y_train_test.shape[0]))}')

#%%
print('Logistic Regression score:')
lr_model.score(X_train_test,y_train_test)

#%%
metrics.accuracy_score(y_train_test, y_pred_train_test)


#%%
# cv_generator = model_selection.KFold(n_splits=5)

lrCV_model = linear_model.LogisticRegressionCV(solver='lbfgs', multi_class='auto', cv=10, n_jobs=-1, random_state=1, max_iter=20)

# lrCV_model = linear_model.LogisticRegressionCV(solver='lbfgs', multi_class='auto', cv=3, n_jobs=-1, random_state=1)
lrCV_model.fit(scaled_train, train_targets)

#%%
y_pred_train_train = lrCV_model.predict(X_train_train)
y_pred_train_test = lrCV_model.predict(X_train_test)
y_pred_train_test.shape

#%%
print('Logistic Regression Cross Validation score for train.train:')
lrCV_model.score(X_train_train,y_train_train)

#%%
print('Logistic Regression Cross Validation score for train.test:')
lrCV_model.score(X_train_test,y_train_test)

#%%
print('Logistic Regression Cross Validation score for train:')
lrCV_model.score(scaled_train, train_targets)

#%%
# import sklearn.model_selection as model_selection
# train, test = model_selection.train_test_split(train_kaggle, random_state=1)

# print(f'train: {train.shape}')
# print(f'test: {test.shape}')

#%%
scaled_test_ids


#%%

y_pred = lrCV_model.predict(scaled_test)
out_df = pandas.DataFrame(y_pred, index=scaled_test_ids, columns=['status_group'])
out_df['status_group'] = out_df['status_group'].replace({1: 'functional', 2: 'functional needs repair', 3: 'non functional'})

#%%
out_df = out_df.reset_index()

#%%
out_df

# set(train_features['date_recorded'].unique()) - set(test_features['date_recorded'].unique())

#%%
# submission = pandas.concat([scaled_test_ids, out_df], axis=1)

#%%
# submission

#%%
out_df.to_csv('./module4/results.csv', index=False)


#%%
