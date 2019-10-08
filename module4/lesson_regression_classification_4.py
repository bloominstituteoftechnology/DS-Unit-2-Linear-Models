#%% [markdown]
# Lambda School Data Science
# 
# *Unit 2, Sprint 1, Module 4*
# 
# ---
# 
# # Regression & Classification, Module 4 (Logistic Regression)
# - do train/validate/test split
# - begin with baselines for classification
# - express and explain the intuition and interpretation of Logistic Regression
# - use sklearn.linear_model.LogisticRegression to fit and interpret Logistic Regression models
# 
# Logistic regression is the baseline for classification models, as well as a handy way to predict probabilities (since those too live in the unit interval). While relatively simple, it is also the foundation for more sophisticated classification techniques such as neural networks (many of which can effectively be thought of as networks of logistic models).
#%% [markdown]
# ### Setup
# 
# You can work locally (follow the [local setup instructions](https://lambdaschool.github.io/ds/unit2/local/)) or on Colab (run the code cell below).
# 
# Libraries:
# - category_encoders 2.0.0
# - numpy
# - pandas
# - scikit-learn

#%%
import os, sys
in_colab = 'google.colab' in sys.modules

# If you're in Colab...
if in_colab:
    # Pull files from Github repo
    os.chdir('/content')
    get_ipython().system('git init .')
    get_ipython().system('git remote add origin https://github.com/LambdaSchool/DS-Unit-2-Regression-Classification.git')
    get_ipython().system('git pull origin master')
    
    # Install required python packages
    get_ipython().system('pip install -r requirements.txt')
    
    # Change into directory for module
    os.chdir('module4')

#%% [markdown]
# # Do train/validate/test split
#%% [markdown]
# ## Overview
#%% [markdown]
# ### Predict Titanic survival 🚢
# 
# Kaggle is a platform for machine learning competitions. [Kaggle has used the Titanic dataset](https://www.kaggle.com/c/titanic/data) for their most popular "getting started" competition. 
# 
# Kaggle splits the data into train and test sets for participants. Let's load both:

#%%
import pandas as pd
train = pd.read_csv('../data/titanic/train.csv')
test = pd.read_csv('../data/titanic/test.csv')

#%% [markdown]
# Notice that the train set has one more column than the test set:

#%%
train.shape, test.shape

#%% [markdown]
# Which column is in train but not test? The target!

#%%
set(train.columns) - set(test.columns)

#%% [markdown]
# ### Why doesn't Kaggle give you the target for the test set?
# 
# #### Rachel Thomas, [How (and why) to create a good validation set](https://www.fast.ai/2017/11/13/validation-sets/)
# 
# > One great thing about Kaggle competitions is that they force you to think about validation sets more rigorously (in order to do well). For those who are new to Kaggle, it is a platform that hosts machine learning competitions. Kaggle typically breaks the data into two sets you can download:
# >
# > 1. a **training set**, which includes the _independent variables,_ as well as the _dependent variable_ (what you are trying to predict).
# >
# > 2. a **test set**, which just has the _independent variables._ You will make predictions for the test set, which you can submit to Kaggle and get back a score of how well you did.
# >
# > This is the basic idea needed to get started with machine learning, but to do well, there is a bit more complexity to understand. **You will want to create your own training and validation sets (by splitting the Kaggle “training” data). You will just use your smaller training set (a subset of Kaggle’s training data) for building your model, and you can evaluate it on your validation set (also a subset of Kaggle’s training data) before you submit to Kaggle.**
# >
# > The most important reason for this is that Kaggle has split the test data into two sets: for the public and private leaderboards. The score you see on the public leaderboard is just for a subset of your predictions (and you don’t know which subset!). How your predictions fare on the private leaderboard won’t be revealed until the end of the competition. The reason this is important is that you could end up overfitting to the public leaderboard and you wouldn’t realize it until the very end when you did poorly on the private leaderboard. Using a good validation set can prevent this. You can check if your validation set is any good by seeing if your model has similar scores on it to compared with on the Kaggle test set. ...
# >
# > Understanding these distinctions is not just useful for Kaggle. In any predictive machine learning project, you want your model to be able to perform well on new data.
#%% [markdown]
# ### 2-way train/test split is not enough
# 
# #### Hastie, Tibshirani, and Friedman, [The Elements of Statistical Learning](http://statweb.stanford.edu/~tibs/ElemStatLearn/), Chapter 7: Model Assessment and Selection
# 
# > If we are in a data-rich situation, the best approach is to randomly divide the dataset into three parts: a training set, a validation set, and a test set. The training set is used to fit the models; the validation set is used to estimate prediction error for model selection; the test set is used for assessment of the generalization error of the final chosen model. Ideally, the test set should be kept in a "vault," and be brought out only at the end of the data analysis. Suppose instead that we use the test-set repeatedly, choosing the model with the smallest test-set error. Then the test set error of the final chosen model will underestimate the true test error, sometimes substantially.
# 
# #### Andreas Mueller and Sarah Guido, [Introduction to Machine Learning with Python](https://books.google.com/books?id=1-4lDQAAQBAJ&pg=PA270)
# 
# > The distinction between the training set, validation set, and test set is fundamentally important to applying machine learning methods in practice. Any choices made based on the test set accuracy "leak" information from the test set into the model. Therefore, it is important to keep a separate test set, which is only used for the final evaluation. It is good practice to do all exploratory analysis and model selection using the combination of a training and a validation set, and reserve the test set for a final evaluation - this is even true for exploratory visualization. Strictly speaking, evaluating more than one model on the test set and choosing the better of the two will result in an overly optimistic estimate of how accurate the model is.
# 
# #### Hadley Wickham, [R for Data Science](https://r4ds.had.co.nz/model-intro.html#hypothesis-generation-vs.hypothesis-confirmation)
# 
# > There is a pair of ideas that you must understand in order to do inference correctly:
# >
# > 1. Each observation can either be used for exploration or confirmation, not both.
# >
# > 2. You can use an observation as many times as you like for exploration, but you can only use it once for confirmation. As soon as you use an observation twice, you’ve switched from confirmation to exploration.
# >
# > This is necessary because to confirm a hypothesis you must use data independent of the data that you used to generate the hypothesis. Otherwise you will be over optimistic. There is absolutely nothing wrong with exploration, but you should never sell an exploratory analysis as a confirmatory analysis because it is fundamentally misleading.
# >
# > If you are serious about doing an confirmatory analysis, one approach is to split your data into three pieces before you begin the analysis.
# 
# 
# #### Sebastian Raschka, [Model Evaluation](https://sebastianraschka.com/blog/2018/model-evaluation-selection-part4.html)
# 
# > Since “a picture is worth a thousand words,” I want to conclude with a figure (shown below) that summarizes my personal recommendations ...
# 
# <img src="https://sebastianraschka.com/images/blog/2018/model-evaluation-selection-part4/model-eval-conclusions.jpg" width="600">
# 
# Usually, we want to do **"Model selection (hyperparameter optimization) _and_ performance estimation."** (The green box in the diagram.)
# 
# Therefore, we usually do **"3-way holdout method (train/validation/test split)"** or **"cross-validation with independent test set."**
#%% [markdown]
# ### What's the difference between Training, Validation, and Testing sets?
# 
# #### Brandon Rohrer, [Training, Validation, and Testing Data Sets](https://end-to-end-machine-learning.teachable.com/blog/146320/training-validation-testing-data-sets)
# 
# > The validation set is for adjusting a model's hyperparameters. The testing data set is the ultimate judge of model performance.
# >
# > Testing data is what you hold out until very last. You only run your model on it once. You don’t make any changes or adjustments to your model after that. ...
#%% [markdown]
# ## Follow Along
# 
# > You will want to create your own training and validation sets (by splitting the Kaggle “training” data).
# 
# Do this, using the [sklearn.model_selection.train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) function:

#%%


#%% [markdown]
# ## Challenge
#%% [markdown]
# For your assignment, you'll begin to participate in a private Kaggle challenge, just for your cohort! 
# 
# You will be provided with data split into 2 sets: training and test. You will create your own training and validation sets, by splitting the Kaggle "training" data, so you'll end up with 3 sets total.
#%% [markdown]
# # Begin with baselines for classification
#%% [markdown]
# ## Overview
#%% [markdown]
# We'll begin with the **majority class baseline.**
# 
# [Will Koehrsen](https://twitter.com/koehrsen_will/status/1088863527778111488)
# 
# > A baseline for classification can be the most common class in the training dataset.
# 
# [*Data Science for Business*](https://books.google.com/books?id=4ZctAAAAQBAJ&pg=PT276), Chapter 7.3: Evaluation, Baseline Performance, and Implications for Investments in Data
# 
# > For classification tasks, one good baseline is the _majority classifier,_ a naive classifier that always chooses the majority class of the training dataset (see Note: Base rate in Holdout Data and Fitting Graphs). This may seem like advice so obvious it can be passed over quickly, but it is worth spending an extra moment here. There are many cases where smart, analytical people have been tripped up in skipping over this basic comparison. For example, an analyst may see a classification accuracy of 94% from her classifier and conclude that it is doing fairly well—when in fact only 6% of the instances are positive. So, the simple majority prediction classifier also would have an accuracy of 94%. 
#%% [markdown]
# ## Follow Along
#%% [markdown]
# Determine majority class

#%%


#%% [markdown]
# What if we guessed the majority class for every prediction?

#%%


#%% [markdown]
# #### Use a classification metric: accuracy
# 
# [Classification metrics are different from regression metrics!](https://scikit-learn.org/stable/modules/model_evaluation.html)
# - Don't use _regression_ metrics to evaluate _classification_ tasks.
# - Don't use _classification_ metrics to evaluate _regression_ tasks.
# 
# [Accuracy](https://scikit-learn.org/stable/modules/model_evaluation.html#accuracy-score) is a common metric for classification. Accuracy is the ["proportion of correct classifications"](https://en.wikipedia.org/wiki/Confusion_matrix): the number of correct predictions divided by the total number of predictions.
#%% [markdown]
# What is the baseline accuracy if we guessed the majority class for every prediction?

#%%



#%%


#%% [markdown]
# ## Challenge
#%% [markdown]
# In your Kaggle challenge, you'll begin with the majority class baseline. How quickly can you beat this baseline?
#%% [markdown]
# # Express and explain the intuition and interpretation of Logistic Regression
# 
#%% [markdown]
# ## Overview
# 
# To help us get an intuition for *Logistic* Regression, let's start by trying *Linear* Regression instead, and see what happens...
#%% [markdown]
# ## Follow Along
#%% [markdown]
# ### Linear Regression?

#%%
train.describe()


#%%
# 1. Import estimator class
from sklearn.linear_model import LinearRegression

# 2. Instantiate this class
linear_reg = LinearRegression()

# 3. Arrange X feature matrices (already did y target vectors)
features = ['Pclass', 'Age', 'Fare']
X_train = train[features]
X_val = val[features]

# Impute missing values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer()
X_train_imputed = imputer.fit_transform(X_train)
X_val_imputed = imputer.transform(X_val)

# 4. Fit the model
linear_reg.fit(X_train_imputed, y_train)

# 5. Apply the model to new data.
# The predictions look like this ...
linear_reg.predict(X_val_imputed)


#%%
# Get coefficients
pd.Series(linear_reg.coef_, features)


#%%
test_case = [[1, 5, 500]]  # 1st class, 5-year old, Rich
linear_reg.predict(test_case)

#%% [markdown]
# ### Logistic Regression!

#%%
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(solver='lbfgs')
log_reg.fit(X_train_imputed, y_train)
print('Validation Accuracy', log_reg.score(X_val_imputed, y_val))


#%%
# The predictions look like this
log_reg.predict(X_val_imputed)


#%%
log_reg.predict(test_case)


#%%
log_reg.predict_proba(test_case)


#%%
# What's the math?
log_reg.coef_


#%%
log_reg.intercept_


#%%
# The logistic sigmoid "squishing" function, implemented to accept numpy arrays
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.e**(-x))


#%%
sigmoid(log_reg.intercept_ + np.dot(log_reg.coef_, np.transpose(test_case)))

#%% [markdown]
# So, clearly a more appropriate model in this situation! For more on the math, [see this Wikipedia example](https://en.wikipedia.org/wiki/Logistic_regression#Probability_of_passing_an_exam_versus_hours_of_study).
#%% [markdown]
# # Use sklearn.linear_model.LogisticRegression to fit and interpret Logistic Regression models
#%% [markdown]
# ## Overview
# 
# Now that we have more intuition and interpretation of Logistic Regression, let's use it within a realistic, complete scikit-learn workflow, with more features and transformations.
#%% [markdown]
# ## Follow Along
# 
# Select these features: `['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']`
# 
# (Why shouldn't we include the `Name` or `Ticket` features? What would happen here?) 
# 
# Fit this sequence of transformers & estimator:
# 
# - [category_encoders.one_hot.OneHotEncoder](https://contrib.scikit-learn.org/categorical-encoding/onehot.html)
# - [sklearn.impute.SimpleImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html)
# - [sklearn.preprocessing.StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
# - [sklearn.linear_model.LogisticRegressionCV](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html)
# 
# Get validation accuracy.

#%%


#%% [markdown]
# Plot coefficients:

#%%


#%% [markdown]
# Generate [Kaggle](https://www.kaggle.com/c/titanic) submission:

#%%


#%% [markdown]
# ## Challenge
# 
# You'll use Logistic Regression for your first model in our Kaggle challenge!
#%% [markdown]
# # Review
# 
# For your assignment, make your first submission to our Kaggle challenge. The assignment notebook has code to load the data, and more details about the instructions:
# 
# - [Sign up for a Kaggle account](https://www.kaggle.com/), if you don’t already have one. Go to our Kaggle InClass competition website. You will be given the URL in Slack. Go to the Rules page. Accept the rules of the competition.
# - Do train/validate/test split with the Tanzania Waterpumps data.
# - Begin with baselines for classification.
# - Use scikit-learn for logistic regression.
# - Get your validation accuracy score.
# - Submit your predictions to our Kaggle competition. (Go to our Kaggle InClass competition webpage. Use the blue **Submit Predictions** button to upload your CSV file. Or you can use the Kaggle API to submit your predictions.)
#%% [markdown]
# # Sources
# - Brandon Rohrer, [Training, Validation, and Testing Data Sets](https://end-to-end-machine-learning.teachable.com/blog/146320/training-validation-testing-data-sets)
# - Hadley Wickham, [R for Data Science](https://r4ds.had.co.nz/model-intro.html#hypothesis-generation-vs.hypothesis-confirmation), Hypothesis generation vs. hypothesis confirmation
# - Hastie, Tibshirani, and Friedman, [The Elements of Statistical Learning](http://statweb.stanford.edu/~tibs/ElemStatLearn/), Chapter 7: Model Assessment and Selection
# - Mueller and Guido, [Introduction to Machine Learning with Python](https://books.google.com/books?id=1-4lDQAAQBAJ&pg=PA270), Chapter 5.2.2: The Danger of Overfitting the Parameters and the Validation Set
# - Provost and Fawcett, [Data Science for Business](https://books.google.com/books?id=4ZctAAAAQBAJ&pg=PT276), Chapter 7.3: Evaluation, Baseline Performance, and Implications for Investments in Data
# - Rachel Thomas, [How (and why) to create a good validation set](https://www.fast.ai/2017/11/13/validation-sets/)
# - Sebastian Raschka, [Model Evaluation](https://sebastianraschka.com/blog/2018/model-evaluation-selection-part4.html)
# - Will Koehrsen, ["A baseline for classification can be the most common class in the training dataset."](https://twitter.com/koehrsen_will/status/1088863527778111488)

