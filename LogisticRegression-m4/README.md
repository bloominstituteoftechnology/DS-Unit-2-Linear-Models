Logistic Regression - 214: Used Titanic dataset to predict passenger survival. This was a classification problem with numerical and categorical features. The dataset was split into training, validation and test data. Training to build the model, validation to estimate the test score and fine tune the model, and test data only used once to get the test score, but not used for tuning the model. The baseline model was simply the majority class which yielded an accuracy score of %60. Initially used three numeric features, Pclass, Age, Fare. Since regression model does not handle missing values, train data was fit_transormed, and validation data was transformed with SimpleImputer() to replace the unknown values with the mean. The imputed input matrics was fit to LinearRegression() and LogisticRegression(), both yielding %73 accuracy score on validation data. Sigmoid function is used to generate predict_proba values for logistic regression model. We used 0.5 as between 0 and 1 to decide the survival class for LinearRegression() numerical predicted results. In 223 validation observations there was only one observation with different prediction for Linear regressor vs. logistic one. LinearRegression() does not have a predict_proba method. However applying sigmoid function to linear regression predicted label yields different values compared to logistic regression predict_proba() values. This could be due to different solver used as the coefficients are also different. Used more features, both numerical and categorical, ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'], first fit_transform() with OneHotEncoder() to convert the categorical features into numeric, while treating missing values as another value for encoding. Next fit_transform() with SimpleImputer(). This would replace all missing values of original numerical features with mean strategy. Next used StandardScaler() to normalize the mean and standard deviation of dataset before fit() it with LogisticRegression. Validation score raises to %81. Coefficients are all regularized with no extremely low or high values.
214a: used dataset of 400+ burrito reviews (https://srcole.github.io/100burritos/). Based of column “Overall” rating created a target label “Great” to be used for a binary classification. As a part of data wrangling rows with Nan in Overall ratings are dropped and any rating above 4 is labeled True for “Great” column. Different types of Burritos are consolidated into 5 categories. The categorical columns with high cardinalities are dropped. Later the selected features are split based on date column into train, validation and test sets and each set divided into X and y. A pipeline consisting of OneHotEncoder(), SimpleImputer(), SelectKBest(), and finally LogisticRegression() was made to train the model with the .fit() method. .predict_proba(), .score(), and .predict() methods were used to evaluate the model accuracy.
Libraries:




import category_encoders as ce
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

