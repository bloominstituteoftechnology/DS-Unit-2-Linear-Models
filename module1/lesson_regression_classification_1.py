#%% [markdown]
# Lambda School Data Science
# 
# *Unit 2, Sprint 1, Module 1*
# 
# ---
#%% [markdown]
# # Regression & Classification, Module 1
# 
# - Begin with baselines for regression
# - Use scikit-learn to fit a linear regression
# - Explain the coefficients from a linear regression
#%% [markdown]
# Brandon Rohrer wrote a good blog post, [“What questions can machine learning answer?”](https://brohrer.github.io/five_questions_data_science_answers.html)
# 
# We’ll focus on two of these questions in Unit 2. These are both types of “supervised learning.”
# 
# - “How Much / How Many?” (Regression)
# - “Is this A or B?” (Classification)
# 
# This unit, you’ll build supervised learning models with “tabular data” (data in tables, like spreadsheets). Including, but not limited to:
# 
# - Predict New York City real estate prices <-- **Today, we'll start this!**
# - Predict which water pumps in Tanzania need repairs
# - Choose your own labeled, tabular dataset, train a predictive model, and publish a blog post or web app with visualizations to explain your model!
#%% [markdown]
# ### Setup
# 
# You can work locally (follow the [local setup instructions](https://lambdaschool.github.io/ds/unit2/local/)) or on Colab (run the code cell below).
# 
# Libraries:
# 
# - ipywidgets
# - pandas
# - plotly 4.1.1
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
    os.chdir('module1')


#%%
# Ignore this Numpy warning when using Plotly Express:
# FutureWarning: Method .ptp is deprecated and will be removed in a future version. Use numpy.ptp instead.
import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning, module='numpy')

#%% [markdown]
# # Begin with baselines for regression
#%% [markdown]
# ## Overview
#%% [markdown]
# ### Predict how much a NYC condo costs 🏠💸
# 
# Regression models output continuous numbers, so we can use regression to answer questions like "How much?" or "How many?" 
# 
# Often, the question is "How much will this cost? How many dollars?"
#%% [markdown]
# For example, here's a fun YouTube video, which we'll use as our scenario for this lesson:
# 
# [Amateurs & Experts Guess How Much a NYC Condo With a Private Terrace Costs](https://www.youtube.com/watch?v=JQCctBOgH9I)
# 
# > Real Estate Agent Leonard Steinberg just sold a pre-war condo in New York City's Tribeca neighborhood. We challenged three people - an apartment renter, an apartment owner and a real estate expert - to try to guess how much the apartment sold for. Leonard reveals more and more details to them as they refine their guesses.
#%% [markdown]
# The condo from the video is **1,497 square feet**, built in 1852, and is in a desirable neighborhood. According to the real estate agent, _"Tribeca is known to be one of the most expensive ZIP codes in all of the United States of America."_
# 
# How can we guess what this condo sold for? Let's look at 3 methods:
# 
# 1. Heuristics
# 2. Descriptive Statistics
# 3. Predictive Model 
#%% [markdown]
# ## Follow Along
#%% [markdown]
# ### 1. Heuristics
# 
# Heuristics are "rules of thumb" that people use to make decisions and judgments. The video participants discussed their heuristics:
# 
# 
# 
#%% [markdown]
# **Participant 1**, Chinwe, is a real estate amateur. She rents her apartment in New York City. Her first guess was \$8 million, and her final guess was \$15 million.
# 
# [She said](https://youtu.be/JQCctBOgH9I?t=465), _"People just go crazy for numbers like 1852. You say **'pre-war'** to anyone in New York City, they will literally sell a kidney. They will just give you their children."_
#%% [markdown]
# **Participant 3**, Pam, is an expert. She runs a real estate blog. Her first guess was \$1.55 million, and her final guess was \$2.2 million.
# 
# [She explained](https://youtu.be/JQCctBOgH9I?t=280) her first guess: _"I went with a number that I think is kind of the going rate in the location, and that's **a thousand bucks a square foot.**"_
# 
#%% [markdown]
# **Participant 2**, Mubeen, is between the others in his expertise level. He owns his apartment in New York City. His first guess was \$1.7 million, and his final guess was also \$2.2 million.
#%% [markdown]
# ### 2. Descriptive Statistics
#%% [markdown]
# We can use data to try to do better than these heuristics. How much have other Tribeca condos sold for?
# 
# Let's answer this question with a relevant dataset, containing most of the single residential unit, elevator apartment condos sold in Tribeca, from January through April 2019.
# 
# We can get descriptive statistics for the dataset's `SALE_PRICE` column.
# 
# How many condo sales are in this dataset? What was the average sale price? The median? Minimum? Maximum?

#%%
import pandas as pd
df = pd.read_csv('../data/condos/tribeca.csv')
pd.options.display.float_format = '{:,.0f}'.format
df['SALE_PRICE'].describe()

#%% [markdown]
# On average, condos in Tribeca have sold for \$3.9 million. So that could be a reasonable first guess.
# 
# In fact, here's the interesting thing: **we could use this one number as a "prediction", if we didn't have any data except for sales price...** 
# 
# Imagine we didn't have any any other information about condos, then what would you tell somebody? If you had some sales prices like this but you didn't have any of these other columns. If somebody asked you, "How much do you think a condo in Tribeca costs?"
# 
# You could say, "Well, I've got 90 sales prices here, and I see that on average they cost \$3.9 million."
# 
# So we do this all the time in the real world. We use descriptive statistics for prediction. And that's not wrong or bad, in fact **that's where you should start. This is called the _mean baseline_.**
#%% [markdown]
# **Baseline** is an overloaded term, with multiple meanings:
# 
# 1. [**The score you'd get by guessing**](https://twitter.com/koehrsen_will/status/1088863527778111488)
# 2. [**Fast, first models that beat guessing**](https://blog.insightdatascience.com/always-start-with-a-stupid-model-no-exceptions-3a22314b9aaa) 
# 3. **Complete, tuned "simpler" model** (Simpler mathematically, computationally. Or less work for you, the data scientist.)
# 4. **Minimum performance that "matters"** to go to production and benefit your employer and the people you serve.
# 5. **Human-level performance** 
# 
# Baseline type #1 is what we're doing now.
# 
# Linear models can be great for #2, 3, 4, and [sometimes even #5 too!](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.188.5825)
#%% [markdown]
# ---
# 
# Let's go back to our mean baseline for Tribeca condos. 
# 
# If we just guessed that every Tribeca condo sold for \$3.9 million, how far off would we be, on average?

#%%
guess = df['SALE_PRICE'].mean()
errors = guess - df['SALE_PRICE']
mean_absolute_error = errors.abs().mean()
print(f'If we just guessed every Tribeca condo sold for ${guess:,.0f},')
print(f'we would be off by ${mean_absolute_error:,.0f} on average.')

#%% [markdown]
# That sounds like a lot of error! 
# 
# But fortunately, we can do better than this first baseline — we can use more data. For example, the condo's size.
# 
# Could sale price be **dependent** on square feet? To explore this relationship, let's make a scatterplot, using [Plotly Express](https://plot.ly/python/plotly-express/):

#%%
import plotly.express as px
px.scatter(df, x='GROSS_SQUARE_FEET', y='SALE_PRICE')

#%% [markdown]
# ### 3. Predictive Model
# 
# To go from a _descriptive_ [scatterplot](https://www.plotly.express/plotly_express/#plotly_express.scatter) to a _predictive_ regression, just add a _line of best fit:_

#%%


#%% [markdown]
# Roll over the Plotly regression line to see its equation and predictions for sale price, dependent on gross square feet.
# 
# Linear Regression helps us **interpolate.** For example, in this dataset, there's a gap between 4016 sq ft and 4663 sq ft. There were no 4300 sq ft condos sold, but what price would you predict, using this line of best fit?
# 
# Linear Regression also helps us **extrapolate.** For example, in this dataset, there were no 6000 sq ft condos sold, but what price would you predict?
#%% [markdown]
# The line of best fit tries to summarize the relationship between our x variable and y variable in a way that enables us to use the equation for that line to make predictions.
# 
# 
# 
# 
#%% [markdown]
# **Synonyms for "y variable"**
# 
# - **Dependent Variable**
# - Response Variable
# - Outcome Variable 
# - Predicted Variable
# - Measured Variable
# - Explained Variable
# - **Label**
# - **Target**
#%% [markdown]
# **Synonyms for "x variable"**
# 
# - **Independent Variable**
# - Explanatory Variable
# - Regressor
# - Covariate
# - **Feature**
# 
#%% [markdown]
# The bolded terminology will be used most often by your instructors this unit.
#%% [markdown]
# ## Challenge
# 
# In your assignment, you will practice how to begin with baselines for regression, using a new dataset!
#%% [markdown]
# # Use scikit-learn to fit a linear regression
#%% [markdown]
# ## Overview
#%% [markdown]
# We can use visualization libraries to do simple linear regression ("simple" means there's only one independent variable). 
# 
# But during this unit, we'll usually use the scikit-learn library for predictive models, and we'll usually have multiple independent variables.
#%% [markdown]
# In [_Python Data Science Handbook,_ Chapter 5.2: Introducing Scikit-Learn](https://jakevdp.github.io/PythonDataScienceHandbook/05.02-introducing-scikit-learn.html#Basics-of-the-API), Jake VanderPlas explains **how to structure your data** for scikit-learn:
# 
# > The best way to think about data within Scikit-Learn is in terms of tables of data. 
# >
# > ![](https://jakevdp.github.io/PythonDataScienceHandbook/figures/05.02-samples-features.png)
# >
# >The features matrix is often stored in a variable named `X`. The features matrix is assumed to be two-dimensional, with shape `[n_samples, n_features]`, and is most often contained in a NumPy array or a Pandas `DataFrame`.
# >
# >We also generally work with a label or target array, which by convention we will usually call `y`. The target array is usually one dimensional, with length `n_samples`, and is generally contained in a NumPy array or Pandas `Series`. The target array may have continuous numerical values, or discrete classes/labels. 
# >
# >The target array is the quantity we want to _predict from the data:_ in statistical terms, it is the dependent variable. 
#%% [markdown]
# VanderPlas also lists a **5 step process** for scikit-learn's "Estimator API":
# 
# > Every machine learning algorithm in Scikit-Learn is implemented via the Estimator API, which provides a consistent interface for a wide range of machine learning applications.
# >
# > Most commonly, the steps in using the Scikit-Learn estimator API are as follows:
# >
# > 1. Choose a class of model by importing the appropriate estimator class from Scikit-Learn.
# > 2. Choose model hyperparameters by instantiating this class with desired values.
# > 3. Arrange data into a features matrix and target vector following the discussion above.
# > 4. Fit the model to your data by calling the `fit()` method of the model instance.
# > 5. Apply the Model to new data: For supervised learning, often we predict labels for unknown data using the `predict()` method.
# 
# Let's try it!
#%% [markdown]
# ## Follow Along
# 
# Follow the 5 step process, and refer to [Scikit-Learn LinearRegression documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html).

#%%
# 1. Import the appropriate estimator class from Scikit-Learn


# 2. Instantiate this class


# 3. Arrange X features matrix & y target vector


# 4. Fit the model


# 5. Apply the model to new data

#%% [markdown]
# So, we used scikit-learn to fit a linear regression, and predicted the sales price for a 1,497 square foot Tribeca condo, like the one from the video.
# 
# Now, what did that condo actually sell for? ___The final answer is revealed in [the video at 12:28](https://youtu.be/JQCctBOgH9I?t=748)!___

#%%


#%% [markdown]
# What was the error for our prediction, versus the video participants?
# 
# Let's use [scikit-learn's mean absolute error function](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html).

#%%
chinwe_final_guess = [15000000]
mubeen_final_guess = [2200000]
pam_final_guess = [2200000]


#%%


#%% [markdown]
# This [diagram](https://ogrisel.github.io/scikit-learn.org/sklearn-tutorial/tutorial/text_analytics/general_concepts.html#supervised-learning-model-fit-x-y) shows what we just did! Don't worry about understanding it all now. But can you start to match some of these boxes/arrows to the corresponding lines of code from above?
# 
# <img src="https://ogrisel.github.io/scikit-learn.org/sklearn-tutorial/_images/plot_ML_flow_chart_12.png" width="75%">
#%% [markdown]
# Wait, are we saying that *linear regression* could be considered a *machine learning algorithm*? Maybe it depends? What do you think? We'll discuss throughout this unit.
#%% [markdown]
# ## Challenge
# 
# In your assignment, you will use scikit-learn for linear regression with one feature. For a stretch goal, you can do linear regression with two or more features.
#%% [markdown]
# # Explain the coefficients from a linear regression
#%% [markdown]
# ## Overview
# 
# What pattern did the model "learn", about the relationship between square feet & price?
#%% [markdown]
# ## Follow Along
#%% [markdown]
# To help answer this question, we'll look at the  `coef_` and `intercept_` attributes of the `LinearRegression` object. (Again, [here's the documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html).)
# 

#%%


#%% [markdown]
# We can repeatedly apply the model to new/unknown data, and explain the coefficient:

#%%
def predict(square_feet):
    y_pred = model.predict([[square_feet]])
    estimate = y_pred[0]
    coefficient = model.coef_[0]
    result = f'${estimate:,.0f} estimated price for {square_feet:,.0f} square foot condo in Tribeca. '
    explanation = f'In this linear regression, each additional square foot adds ${coefficient:,.0f}.'
    return result + explanation

predict(1497)


#%%
# What does the model predict for low square footage?
predict(500)


#%%
# For high square footage?
predict(10000)


#%%


#%% [markdown]
# ## Challenge
# 
# In your assignment, you will define a function to make new predictions and explain the model coefficient.
#%% [markdown]
# # Review
#%% [markdown]
# You'll practice these objectives when you do your assignment:
# 
# - Begin with baselines for regression
# - Use scikit-learn to fit a linear regression
# - Make new predictions and explain coefficients
#%% [markdown]
# You'll use another New York City real estate dataset. You'll predict how much it costs to rent an apartment, instead of how much it costs to buy a condo.
# 
# You've been provided with a separate notebook for your assignment, which has all the instructions and stretch goals. Good luck and have fun!
#%% [markdown]
# # Sources
# 
# #### NYC Real Estate
# - Video: [Amateurs & Experts Guess How Much a NYC Condo With a Private Terrace Costs](https://www.youtube.com/watch?v=JQCctBOgH9I)
# - Data: [NYC OpenData: NYC Citywide Rolling Calendar Sales](https://data.cityofnewyork.us/dataset/NYC-Citywide-Rolling-Calendar-Sales/usep-8jbt)
# - Glossary: [NYC Department of Finance: Rolling Sales Data](https://www1.nyc.gov/site/finance/taxes/property-rolling-sales-data.page)
# 
# #### Baselines
# - Will Koehrsen, ["One of the most important steps in a machine learning project is establishing a common sense baseline..."](https://twitter.com/koehrsen_will/status/1088863527778111488)
# - Emmanuel Ameisen, [Always start with a stupid model, no exceptions](https://blog.insightdatascience.com/always-start-with-a-stupid-model-no-exceptions-3a22314b9aaa)
# - Robyn M. Dawes, [The robust beauty of improper linear models in decision making](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.188.5825)
# 
# #### Plotly Express
# - [Plotly Express](https://plot.ly/python/plotly-express/) examples
# - [plotly_express.scatter](https://www.plotly.express/plotly_express/#plotly_express.scatter) docs
# 
# #### Scikit-Learn
# - Jake VanderPlas, [_Python Data Science Handbook,_ Chapter 5.2: Introducing Scikit-Learn](https://jakevdp.github.io/PythonDataScienceHandbook/05.02-introducing-scikit-learn.html#Basics-of-the-API)
# - Olvier Grisel, [Diagram](https://ogrisel.github.io/scikit-learn.org/sklearn-tutorial/tutorial/text_analytics/general_concepts.html#supervised-learning-model-fit-x-y)
# - [sklearn.linear_model.LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
# - [sklearn.metrics.mean_absolute_error](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html)

