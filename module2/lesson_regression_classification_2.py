#%% [markdown]
# Lambda School Data Science
# 
# *Unit 2, Sprint 1, Module 2*
# 
# ---
#%% [markdown]
# # Regression & Classification, Module 2
# - Do train/test split
# - Use scikit-learn to fit a multiple regression
# - Understand how ordinary least squares regression minimizes the sum of squared errors
# - Define overfitting/underfitting and the bias/variance tradeoff
#%% [markdown]
# ### Setup
# 
# You can work locally (follow the [local setup instructions](https://lambdaschool.github.io/ds/unit2/local/)) or on Colab (run the code cell below).
# 
# Libraries:
# - matplotlib
# - numpy
# - pandas
# - plotly 4.1.1
# - scikit-learn

#%%
import os, sys
# in_colab = 'google.colab' in sys.modules

# # If you're in Colab...
# if in_colab:
#     # Pull files from Github repo
#     os.chdir('/content')
#     get_ipython().system('git init .')
#     get_ipython().system('git remote add origin https://github.com/LambdaSchool/DS-Unit-2-Regression-Classification.git')
#     get_ipython().system('git pull origin master')
    
#     # Install required python packages
#     get_ipython().system('pip install -r requirements.txt')
    
#     # Change into directory for module
#     os.chdir('module2')


#%%
# Ignore this Numpy warning when using Plotly Express:
# FutureWarning: Method .ptp is deprecated and will be removed in a future version. Use numpy.ptp instead.
import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning, module='numpy')

#%% [markdown]
# # Do train/test split
#%% [markdown]
# ## Overview
#%% [markdown]
# ### Predict Elections! 🇺🇸🗳️
#%% [markdown]
# How could we try to predict the 2020 US Presidential election? 
# 
# According to Douglas Hibbs, a political science and economics professor, you can [explain elections with just two features, "Bread and Peace":](https://douglas-hibbs.com/background-information-on-bread-and-peace-voting-in-us-presidential-elections/)
# 
# > Aggregate two-party vote shares going to candidates of the party holding the presidency during the postwar era are well explained by just two fundamental determinants:
# >
# > (1) Positively by weighted-average growth of per capita real disposable personal income over the term.  
# > (2) Negatively by cumulative US military fatalities (scaled to population) owing to unprovoked, hostile deployments of American armed forces in foreign wars. 
#%% [markdown]
# Let's look at the data that Hibbs collected and analyzed:

#%%
import pandas as pd
df = pd.read_csv('./data/elections/bread_peace_voting.csv')
df

#%% [markdown]
# Data Sources & Definitions
# 
# - 1952-2012: Douglas Hibbs, [2014 lecture at Deakin University Melbourne](http://www.douglas-hibbs.com/HibbsArticles/HIBBS-PRESVOTE-SLIDES-MELBOURNE-Part1-2014-02-26.pdf), Slide 40
# - 2016, Vote Share: [The American Presidency Project](https://www.presidency.ucsb.edu/statistics/elections)
# - 2016, Recent Growth in Personal Incomes: [The 2016 election economy: the "Bread and Peace" model final forecast](https://angrybearblog.com/2016/11/the-2016-election-economy-the-bread-and-peace-model-final-forecast.html)
# - 2016, US Military Fatalities: Assumption that Afghanistan War fatalities in 2012-16 occured at the same rate as 2008-12
# 
# > Fatalities denotes the cumulative number of American military fatalities per millions of US population the in Korea, Vietnam, Iraq and Afghanistan wars during the presidential terms preceding the 1952, 1964, 1968, 1976 and 2004, 2008 and 2012 elections. —[Hibbs](http://www.douglas-hibbs.com/HibbsArticles/HIBBS-PRESVOTE-SLIDES-MELBOURNE-Part1-2014-02-26.pdf), Slide 33
#%% [markdown]
# Here we have data from the 1952-2016 elections. We could make a model to predict 1952-2016 election outcomes — but do we really care about that? 
# 
# No, not really. We already know what happened, we don't need to predict it.
#%% [markdown]
# This is explained in [_An Introduction to Statistical Learning_](http://faculty.marshall.usc.edu/gareth-james/ISL/), Chapter 2.2, Assessing Model Accuracy:
# 
# > In general, we do not really care how well the method works training on the training data. Rather, _we are interested in the accuracy of the predictions that we obtain when we apply our method to previously unseen test data._ Why is this what we care about? 
# >
# > Suppose that we are interested in developing an algorithm to predict a stock’s price based on previous stock returns. We can train the method using stock returns from the past 6 months. But we don’t really care how well our method predicts last week’s stock price. We instead care about how well it will predict tomorrow’s price or next month’s price. 
# >
# > On a similar note, suppose that we have clinical measurements (e.g. weight, blood pressure, height, age, family history of disease) for a number of patients, as well as information about whether each patient has diabetes. We can use these patients to train a statistical learning method to predict risk of diabetes based on clinical measurements. In practice, we want this method to accurately predict diabetes risk for _future patients_ based on their clinical measurements. We are not very interested in whether or not the method accurately predicts diabetes risk for patients used to train the model, since we already know which of those patients have diabetes.
#%% [markdown]
# So, we're really interested in the 2020 election — but we probably don't want to wait until then to evaluate our model.
# 
# There is a way we can estimate now how well our model will generalize in the future. We can't fast-forward time, but we can rewind it...
# 
# We can split our data in **two sets.** For example: 
# 1. **Train** a model on elections before 2008.
# 2. **Test** the model on 2008, 2012, 2016. 
# 
# This "backtesting" helps us estimate how well the model will predict the next elections going forward, starting in 2020.
#%% [markdown]
# This is explained in [_Forecasting,_ Chapter 3.4,](https://otexts.com/fpp2/accuracy.html) Evaluating forecast accuracy:
# 
# > The accuracy of forecasts can only be determined by considering how well a model performs on new data that were not used when fitting the model.
# >
# >When choosing models, it is common practice to separate the available data into two portions, training and test data, where the training data is used to estimate any parameters of a forecasting method and the test data is used to evaluate its accuracy. Because the test data is not used in determining the forecasts, it should provide a reliable indication of how well the model is likely to forecast on new data.
# >
# >![](https://otexts.com/fpp2/fpp_files/figure-html/traintest-1.png)
# >
# >The size of the test set is typically about 20% of the total sample, although this value depends on how long the sample is and how far ahead you want to forecast. The following points should be noted.
# >
# >- A model which fits the training data well will not necessarily forecast well.
# >- A perfect fit can always be obtained by using a model with enough parameters.
# >- Over-fitting a model to data is just as bad as failing to identify a systematic pattern in the data.
# >
# >Some references describe the test set as the “hold-out set” because these data are “held out” of the data used for fitting. Other references call the training set the “in-sample data” and the test set the “out-of-sample data”. We prefer to use “training data” and “test data” in this book.
#%% [markdown]
# ## Follow Along
# 
# Split the data in two sets:
# 1. Train on elections before 2008.
# 2. Test on 2008 and after.

#%%


#%% [markdown]
# How many observations (rows) are in the train set? In the test set?

#%%


#%% [markdown]
# Note that this volume of data is at least two orders of magnitude smaller than we usually want to work with for predictive modeling.
# 
# There are other validation techniques that could be used here, such as [time series cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split), or [leave-one-out cross validation](https://scikit-learn.org/stable/modules/cross_validation.html#leave-one-out-loo) for small datasets. However, for this module, let's start simpler, with train/test split. 
# 
# Using a tiny dataset is intentional here. It's good for learning because we can see all the data at once.
#%% [markdown]
# ## Challenge
# 
# In your assignment, you will do train/test split, based on date.
#%% [markdown]
# # Use scikit-learn to fit a multiple regression
#%% [markdown]
# ## Overview
# 
# We've done train/test split, and we're ready to fit a model. 
# 
# We'll proceed in 3 steps. The first 2 are review from the previous module. The 3rd is new.
# 
# - Begin with baselines (0 features) 
# - Simple regression (1 feature)
# - Multiple regression (2 features)
#%% [markdown]
# ## Follow Along
#%% [markdown]
# ### Begin with baselines (0 features)
#%% [markdown]
# What was the average Incumbent Party Vote Share, in the 1952-2004 elections?

#%%
train = df[df['Year'] <= 2004]
test = df[df['Year'] > 2004]

train['Incumbent Party Vote Share'].mean()

#%% [markdown]
# What if we guessed this number for every election? How far off would this be on average?

#%%
from sklearn.metrics import mean_absolute_error

# Arrange y target vectors
target = 'Incumbent Party Vote Share'
y_train = train[target]
y_test = test[target]

# Get mean baseline
print('Mean Baseline (using 0 features)')
guess = y_train.mean()

# Train Error
y_pred = [guess] * len(y_train)
mae = mean_absolute_error(y_train, y_pred)
print(f'Train Error (1952-2004 elections): {mae:.2f} percentage points')

# Test Error
y_pred = [guess] * len(y_test)
mae = mean_absolute_error(y_test, y_pred)
print(f'Test Error (2008-16 elections): {mae:.2f} percentage points')

#%% [markdown]
# ### Simple regression (1 feature)
#%% [markdown]
# Make a scatterplot of the relationship between 1 feature and the target.
# 
# We'll use an economic feature: Average Recent Growth in Personal Incomes. ("Bread")

#%%
import pandas as pd
import plotly.express as px

px.scatter(
    train,
    x='Average Recent Growth in Personal Incomes',
    y='Incumbent Party Vote Share',
    text='Year',
    title='US Presidential Elections, 1952-2004',
    trendline='ols',  # Ordinary Least Squares
)

#%% [markdown]
# 1952 & 1968 are outliers: The incumbent party got fewer votes than predicted by the regression. What do you think could explain those years? We'll come back to this soon, but first...
#%% [markdown]
# Use scikit-learn to fit the simple regression with one feature.
# 
# Follow the [5 step process](https://jakevdp.github.io/PythonDataScienceHandbook/05.02-introducing-scikit-learn.html#Basics-of-the-API), and refer to [Scikit-Learn LinearRegression documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html).

#%%
# 1. Import the appropriate estimator class from Scikit-Learn
from sklearn.linear_model import LinearRegression

# 2. Instantiate this class
model = LinearRegression()

# 3. Arrange X features matrices (already did y target vectors)
features = ['Average Recent Growth in Personal Incomes']
X_train = train[features]
X_test = test[features]
print(f'Linear Regression, dependent on: {features}')

# 4. Fit the model
model.fit(X_train, y_train)
y_pred = model.predict(X_train)
mae = mean_absolute_error(y_train, y_pred)
print(f'Train Error: {mae:.2f} percentage points')

# 5. Apply the model to new data
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f'Test Error: {mae:.2f} percentage points')

#%% [markdown]
# How does the error compare to the baseline?
#%% [markdown]
# ### Multiple regression (2 features)
#%% [markdown]
# Make a scatterplot of the relationship between 2 features and the target.
# 
# We'll add another feature: US Military Fatalities per Million. ("Peace" or the lack thereof.)
# 
# Rotate the scatterplot to explore the data. What's different about 1952 & 1968?

#%%
px.scatter_3d(
    train,
    x='Average Recent Growth in Personal Incomes', 
    y='US Military Fatalities per Million', 
    z='Incumbent Party Vote Share', 
    text='Year', 
    title='US Presidential Elections, 1952-2004'
)

#%% [markdown]
# Use scikit-learn to fit a multiple regression with two features.

#%%
# TODO: Complete this cell

# Re-arrange X features matrices
features = ['Average Recent Growth in Personal Incomes', 
            'US Military Fatalities per Million']
print(f'Linear Regression, dependent on: {features}')



# Fit the model



# Apply the model to new data


#%% [markdown]
# How does the error compare to the prior model?
#%% [markdown]
# ### Plot the plane of best fit
#%% [markdown]
# For a regression with 1 feature, we plotted the line of best fit in 2D. 
# 
# (There are many ways to do this. Plotly Express's `scatter` function makes it convenient with its `trendline='ols'` parameter.)
# 
# For a regression with 2 features, we can plot the plane of best fit in 3D!
# 
# (Plotly Express has a `scatter_3d` function but it won't plot the plane of best fit for us. But, we can write our own function, with the same "function signature" as the Plotly Express API.)

#%%
import itertools
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression

def regression_3d(df, x, y, z, num=100, **kwargs):
    """
    Visualize linear regression in 3D: 2 features + 1 target
    
    df : Pandas DataFrame
    x : string, feature 1 column in df
    y : string, feature 2 column in df
    z : string, target column in df
    num : integer, number of quantiles for each feature
    """
    
    # Plot data
    fig = px.scatter_3d(df, x, y, z, **kwargs)
    
    # Fit Linear Regression
    features = [x, y]
    target = z
    model = LinearRegression()
    model.fit(df[features], df[target])    
    
    # Define grid of coordinates in the feature space
    xmin, xmax = df[x].min(), df[x].max()
    ymin, ymax = df[y].min(), df[y].max()
    xcoords = np.linspace(xmin, xmax, num)
    ycoords = np.linspace(ymin, ymax, num)
    coords = list(itertools.product(xcoords, ycoords))
    
    # Make predictions for the grid
    predictions = model.predict(coords)
    Z = predictions.reshape(num, num).T
    
    # Plot predictions as a 3D surface (plane)
    fig.add_trace(go.Surface(x=xcoords, y=ycoords, z=Z))
    
    return fig


#%%
regression_3d(
    train,
    x='Average Recent Growth in Personal Incomes', 
    y='US Military Fatalities per Million', 
    z='Incumbent Party Vote Share', 
    text='Year', 
    title='US Presidential Elections, 1952-2004'
)

#%% [markdown]
# Where are 1952 & 1968 in relation to the plane? Which elections are the biggest outliers now?
#%% [markdown]
# Roll over points on the plane to see predicted incumbent party vote share (z axis), dependent on personal income growth (x axis) and military fatatlies per capita (y axis).
#%% [markdown]
# ### Get and interpret coefficients
#%% [markdown]
# During the previous module, we got the simple regression's coefficient and intercept. We plugged these numbers into an equation for the line of best fit, in slope-intercept form: $y = mx + b$
# 
# Let's review this objective, but now for multiple regression.
# 
# What's the equation for the plane of best fit?
# 
# $y = \beta_0 + \beta_1x_1 + \beta_2x_2$
# 
# Can you relate the intercept and coefficients to what you see in the plot above?

#%%
model.intercept_, model.coef_


#%%
beta0 = model.intercept_
beta1, beta2 = model.coef_
print(f'y = {beta0} + {beta1}x1 + {beta2}x2')


#%%
# This is easier to read
print('Intercept', model.intercept_)
coefficients = pd.Series(model.coef_, features)
print(coefficients.to_string())

#%% [markdown]
# One of the coefficients is positive, and the other is negative. What does this mean?
#%% [markdown]
# What does the model predict if income growth=0%, fatalities=0

#%%
model.predict([[0, 0]])

#%% [markdown]
# Income growth = 1% (fatalities = 0)

#%%
model.predict([[1, 0]])

#%% [markdown]
# The difference between these predictions = ? 

#%%
model.predict([[1, 0]]) - model.predict([[0, 0]])

#%% [markdown]
# What if... income growth = 2% (fatalities = 0)

#%%
model.predict([[2, 0]])

#%% [markdown]
# The difference between these predictions = ?

#%%
model.predict([[2, 0]]) - model.predict([[1, 0]])

#%% [markdown]
# What if... (income growth=2%) fatalities = 100

#%%
model.predict([[2, 100]])

#%% [markdown]
# The difference between these predictions = ?

#%%
model.predict([[2, 100]]) - model.predict([[2, 0]])

#%% [markdown]
# What if income growth = 3% (fatalities = 100)

#%%
model.predict([[3, 100]])

#%% [markdown]
# The difference between these predictions = ?

#%%
model.predict([[3, 100]]) - model.predict([[2, 100]])

#%% [markdown]
# What if (income growth = 3%) fatalities = 200

#%%
model.predict([[3, 200]])

#%% [markdown]
# The difference between these predictions = ?

#%%
model.predict([[3, 200]]) - model.predict([[3, 100]])

#%% [markdown]
# ## Challenge
# 
# In your assignment, you'll fit a Linear Regression with at least 2 features.
#%% [markdown]
# # Understand how ordinary least squares regression minimizes the sum of squared errors
#%% [markdown]
# ## Overview
# 
# So far, we've evaluated our models by their absolute error. It's an intuitive metric for regression problems.
# 
# However, ordinary least squares doesn't directly minimize absolute error. Instead, it minimizes squared error.
# 
# 
# 
#%% [markdown]
# In this section, we'll introduce two new regression metrics: 
# 
# - Squared error
# - $R^2$
# 
#%% [markdown]
# We'll demostrate two possible methods to minimize squared error:
# 
# - Guess & check
# - Linear Algebra
#%% [markdown]
# ## Follow Along
#%% [markdown]
# ### Guess & Check
# 
# This function visualizes squared errors. We'll go back to simple regression with 1 feature, because it's much easier to visualize.
# 
# Use the function's m & b parameters to "fit the model" manually. Guess & check what values of m & b minimize squared error.

#%%
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def squared_errors(df, feature, target, m, b):
    """
    Visualize linear regression, with squared errors,
    in 2D: 1 feature + 1 target.
    
    Use the m & b parameters to "fit the model" manually.
    
    df : Pandas DataFrame
    feature : string, feature column in df
    target : string, target column in df
    m : numeric, slope for linear equation
    b : numeric, intercept for linear requation
    """
    
    # Plot data
    fig = plt.figure(figsize=(7,7))
    ax = plt.axes()
    df.plot.scatter(feature, target, ax=ax)
    
    # Make predictions
    x = df[feature]
    y = df[target]
    y_pred = m*x + b
    
    # Plot predictions
    ax.plot(x, y_pred)
    
    # Plot squared errors
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    scale = (xmax-xmin)/(ymax-ymin)
    for x, y1, y2 in zip(x, y, y_pred):
        bottom_left = (x, min(y1, y2))
        height = abs(y1 - y2)
        width = height * scale
        ax.add_patch(Rectangle(xy=bottom_left, width=width, height=height, alpha=0.1))
    
    # Print regression metrics
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    print('Mean Squared Error:', mse)
    print('Root Mean Squared Error:', rmse)
    print('Mean Absolute Error:', mae)
    print('R^2:', r2)

#%% [markdown]
# Here's what the mean baseline looks like:

#%%
feature = 'Average Recent Growth in Personal Incomes'
squared_errors(train, feature, target, m=0, b=y_train.mean())

#%% [markdown]
# Notice that $R^2$ is exactly zero. 
# 
# [$R^2$ represents the proportion of the variance for a dependent variable that is explained by the independent variable(s).](https://en.wikipedia.org/wiki/Coefficient_of_determination)
# 
# The mean baseline uses zero independent variables and explains none of the variance in the dependent variable, so its $R^2$ score is zero.
# 
# The highest possible $R^2$ score is 1. The lowest possible *Train* $R^2$ score with ordinary least squares regression is 0.
# 
# In this demo, it's possible to get a negative Train $R^2$, if you manually set values of m & b that are worse than the mean baseline. But that wouldn't happen in the real world.
# 
# However, in the real world, it _is_ possible to get a negative *Test/Validation* $R^2$. It means that your *Test/Validation* predictions are worse than if you'd constantly predicted the mean of the *Test/Validation* set.
#%% [markdown]
# ---
# 
# Now that we've visualized the squared errors for the mean baseline, let's guess & check some better values for the m & b parameters:

#%%
squared_errors(train, feature, target, m=3, b=46)

#%% [markdown]
# You can run the function repeatedly, with different values for m & b.
# 
# How do you interpret each metric you see?
# 
# - Mean Squared Error
# - Root Mean Squared Error
# - Mean Absolute Error
# - $R^2$
# 
# Does guess & check really get used in machine learning? Sometimes! Some complex functions are hard to minimize, so we use a sophisticated form of guess & check called "gradient descent", which you'll learn about in Unit 4.
# 
# Fortunately, we don't need to use guess & check for ordinary least squares regression. We have a solution, using linear algebra!
# 
#%% [markdown]
# ### Linear Algebra
# 
# The same result that is found by minimizing the sum of the squared errors can be also found through a linear algebra process known as the "Least Squares Solution:"
# 
# \begin{align}
# \hat{\beta} = (X^{T}X)^{-1}X^{T}y
# \end{align}
# 
# Before we can work with this equation in its linear algebra form we have to understand how to set up the matrices that are involved in this equation. 
# 
# #### The $\beta$ vector
# 
# The $\beta$ vector represents all the parameters that we are trying to estimate, our $y$ vector and $X$ matrix values are full of data from our dataset. The $\beta$ vector holds the variables that we are solving for: $\beta_0$ and $\beta_1$
# 
# Now that we have all of the necessary parts we can set them up in the following equation:
# 
# \begin{align}
# y = X \beta + \epsilon
# \end{align}
# 
# Since our $\epsilon$ value represents **random** error we can assume that it will equal zero on average.
# 
# \begin{align}
# y = X \beta
# \end{align}
# 
# The objective now is to isolate the $\beta$ matrix. We can do this by pre-multiplying both sides by "X transpose" $X^{T}$.
# 
# \begin{align}
# X^{T}y =  X^{T}X \beta
# \end{align}
# 
# Since anything times its transpose will result in a square matrix, if that matrix is then an invertible matrix, then we should be able to multiply both sides by its inverse to remove it from the right hand side. (We'll talk tomorrow about situations that could lead to $X^{T}X$ not being invertible.)
# 
# \begin{align}
# (X^{T}X)^{-1}X^{T}y =  (X^{T}X)^{-1}X^{T}X \beta
# \end{align}
# 
# Since any matrix multiplied by its inverse results in the identity matrix, and anything multiplied by the identity matrix is itself, we are left with only $\beta$ on the right hand side:
# 
# \begin{align}
# (X^{T}X)^{-1}X^{T}y = \hat{\beta}
# \end{align}
# 
# We will now call it "beta hat" $\hat{\beta}$ because it now represents our estimated values for $\beta_0$ and $\beta_1$
# 
# #### Lets calculate our $\beta$ parameters with numpy!

#%%
# This is NOT something you'll be tested on. It's just a demo.

# X is a matrix. Add column of constants for fitting the intercept.
def add_constant(X):
    constant = np.ones(shape=(len(X),1))
    return np.hstack((constant, X))
X = add_constant(train[features].values)
print('X')
print(X)

# y is a column vector
y = train[target].values[:, np.newaxis]
print('y')
print(y)

# Least squares solution in code
X_transpose = X.T
X_transpose_X = X_transpose @ X
X_transpose_X_inverse = np.linalg.inv(X_transpose_X)
X_transpose_y = X_transpose @ y
beta_hat = X_transpose_X_inverse @ X_transpose_y

print('Beta Hat')
print(beta_hat)


#%%
# Scikit-learn gave the exact same results!
model.intercept_, model.coef_

#%% [markdown]
# # Define overfitting/underfitting and the bias/variance tradeoff
#%% [markdown]
# ## Overview
#%% [markdown]
# Read [_Python Data Science Handbook,_ Chapter 5.3](https://jakevdp.github.io/PythonDataScienceHandbook/05.03-hyperparameters-and-model-validation.html#The-Bias-variance-trade-off). Jake VanderPlas explains overfitting & underfitting:
# 
# > Fundamentally, the question of "the best model" is about finding a sweet spot in the tradeoff between bias and variance. Consider the following figure, which presents two regression fits to the same dataset:
# > 
# >![](https://jakevdp.github.io/PythonDataScienceHandbook/figures/05.03-bias-variance-2.png)
# >
# > The model on the left attempts to find a straight-line fit through the data. Because the data are intrinsically more complicated than a straight line, the straight-line model will never be able to describe this dataset well. Such a model is said to _underfit_ the data: that is, it does not have enough model flexibility to suitably account for all the features in the data; another way of saying this is that the model has high _bias_.
# >
# > The model on the right attempts to fit a high-order polynomial through the data. Here the model fit has enough flexibility to nearly perfectly account for the fine features in the data, but even though it very accurately describes the training data, its precise form seems to be more reflective of the particular noise properties of the data rather than the intrinsic properties of whatever process generated that data. Such a model is said to _overfit_ the data: that is, it has so much model flexibility that the model ends up accounting for random errors as well as the underlying data distribution; another way of saying this is that the model has high _variance_.
#%% [markdown]
# VanderPlas goes on to connect these concepts to the "bias/variance tradeoff":
# 
# > From the scores associated with these two models, we can make an observation that holds more generally:
# >
# >- For high-bias models, the performance of the model on the validation set is similar to the performance on the training set.
# >
# >- For high-variance models, the performance of the model on the validation set is far worse than the performance on the training set.
# >
# > If we imagine that we have some ability to tune the model complexity, we would expect the training score and validation score to behave as illustrated in the following figure:
# >
# >![](https://jakevdp.github.io/PythonDataScienceHandbook/figures/05.03-validation-curve.png)
# >
# > The diagram shown here is often called a validation curve, and we see the following essential features:
# >
# >- The training score is everywhere higher than the validation score. This is generally the case: the model will be a better fit to data it has seen than to data it has not seen.
# >- For very low model complexity (a high-bias model), the training data is under-fit, which means that the model is a poor predictor both for the training data and for any previously unseen data.
# >- For very high model complexity (a high-variance model), the training data is over-fit, which means that the model predicts the training data very well, but fails for any previously unseen data.
# >- For some intermediate value, the validation curve has a maximum. This level of complexity indicates a suitable trade-off between bias and variance.
# >
# >The means of tuning the model complexity varies from model to model.
#%% [markdown]
# So far, our only "means of tuning the model complexity" has been selecting one feature or two features for our linear regression models. But we'll quickly start to select more features, and more complex models, with more "hyperparameters."
# 
# This is just a first introduction to underfitting & overfitting. We'll continue to learn about this topic all throughout this unit.
#%% [markdown]
# ## Follow Along
#%% [markdown]
# Let's make our own Validation Curve, by tuning a new type of model complexity: polynomial degrees in a linear regression.
#%% [markdown]
# Go back to the the NYC Tribeca condo sales data

#%%
# Read NYC Tribeca condo sales data, from first 4 months of 2019.
# Dataset has 90 rows, 9 columns.
df = pd.read_csv('./data/condos/tribeca.csv')
assert df.shape == (90, 9)

# Arrange X features matrix & y target vector
features = ['GROSS_SQUARE_FEET']
target = 'SALE_PRICE'
X = df[features]
y = df[target]

#%% [markdown]
# Do random [train/test split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)

#%%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=11)

#%% [markdown]
# Repeatedly fit increasingly complex models, and keep track of the scores

#%%
from IPython.display import display, HTML
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


# Credit for PolynomialRegression: Jake VanderPlas, Python Data Science Handbook, Chapter 5.3
# https://jakevdp.github.io/PythonDataScienceHandbook/05.03-hyperparameters-and-model-validation.html#Validation-curves-in-Scikit-Learn
def PolynomialRegression(degree=2, **kwargs):
    return make_pipeline(PolynomialFeatures(degree), 
                         LinearRegression(**kwargs))


polynomial_degrees = range(1, 10, 2)
train_r2s = []
test_r2s = []

for degree in polynomial_degrees:
    model = PolynomialRegression(degree)
    display(HTML(f'Polynomial degree={degree}'))
    
    model.fit(X_train, y_train)
    train_r2 = model.score(X_train, y_train)
    test_r2 = model.score(X_test, y_test)
    display(HTML(f'<b style="color: blue">Train R2 {train_r2:.2f}</b>'))
    display(HTML(f'<b style="color: red">Test R2 {test_r2:.2f}</b>'))

    plt.scatter(X_train, y_train, color='blue', alpha=0.5)
    plt.scatter(X_test, y_test, color='red', alpha=0.5)
    plt.xlabel(features)
    plt.ylabel(target)
    
    x_domain = np.linspace(X.min(), X.max())
    curve = model.predict(x_domain)
    plt.plot(x_domain, curve, color='blue')
    plt.show()
    display(HTML('<hr/>'))
    
    train_r2s.append(train_r2)
    test_r2s.append(test_r2)
    
display(HTML('Validation Curve'))
plt.plot(polynomial_degrees, train_r2s, color='blue', label='Train')
plt.plot(polynomial_degrees, test_r2s, color='red', label='Test')
plt.xlabel('Model Complexity (Polynomial Degree)')
plt.ylabel('R^2 Score')
plt.legend()
plt.show()

#%% [markdown]
# As model complexity increases, what happens to Train $R^2$ and Test $R^2$?
#%% [markdown]
# # Review
# 
# In your assignment, you'll continue to **predict how much it costs to rent an apartment in NYC,** using the dataset from renthop.com.
# 
# 
# - Do train/test split. Use data from April & May 2016 to train. Use data from June 2016 to test.
# - Engineer at least two new features.
# - Fit a linear regression model with at least two features.
# - Get the model's coefficients and intercept.
# - Get regression metrics RMSE, MAE, and $R^2$, for both the train and test sets.
# 
# You've been provided with a separate notebook for your assignment, which has all the instructions and stretch goals. What's the best test MAE you can get? Share your score and features used with your cohort on Slack!
#%% [markdown]
# # Sources
# 
# #### Train/Test Split
# - James, Witten, Hastie, Tibshirani, [_An Introduction to Statistical Learning_](http://faculty.marshall.usc.edu/gareth-james/ISL/), Chapter 2.2, Assessing Model Accuracy
# - Hyndman, Athanasopoulos, [_Forecasting,_ Chapter 3.4,](https://otexts.com/fpp2/accuracy.html) Evaluating forecast accuracy
# 
# #### Bias-Variance Tradeoff
# - Jake VanderPlas, [_Python Data Science Handbook,_ Chapter 5.3](https://jakevdp.github.io/PythonDataScienceHandbook/05.03-hyperparameters-and-model-validation.html#The-Bias-variance-trade-off), Hyperparameters and Model Validation
# 
# 
# #### "Bread and Peace" Background
# - Douglas Hibbs, [Background Information on the ‘Bread and Peace’ Model of Voting in Postwar US Presidential Elections](https://douglas-hibbs.com/background-information-on-bread-and-peace-voting-in-us-presidential-elections/)
# 
# 
# #### "Bread and Peace" Data Sources & Definitions
# - 1952-2012: Douglas Hibbs, [2014 lecture at Deakin University Melbourne](http://www.douglas-hibbs.com/HibbsArticles/HIBBS-PRESVOTE-SLIDES-MELBOURNE-Part1-2014-02-26.pdf), Slide 40
# - 2016, Vote Share: [The American Presidency Project](https://www.presidency.ucsb.edu/statistics/elections)
# - 2016, Recent Growth in Personal Incomes: [The 2016 election economy: the "Bread and Peace" model final forecast](https://angrybearblog.com/2016/11/the-2016-election-economy-the-bread-and-peace-model-final-forecast.html)
# - 2016, US Military Fatalities: Assumption that Afghanistan War fatalities in 2012-16 occured at the same rate as 2008-12
# 
# > Fatalities denotes the cumulative number of American military fatalities per millions of US population the in Korea, Vietnam, Iraq and Afghanistan wars during the presidential terms preceding the 1952, 1964, 1968, 1976 and 2004, 2008 and 2012 elections. —[Hibbs](http://www.douglas-hibbs.com/HibbsArticles/HIBBS-PRESVOTE-SLIDES-MELBOURNE-Part1-2014-02-26.pdf), Slide 33

