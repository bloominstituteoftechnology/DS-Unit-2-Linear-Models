Used NYC apartment rental data. Converted “created” column to datetime format to be used for splitting the data into training and validation subsets. Feature engineered new columns as “pets”, “rooms”, and “perk_count”. removed high cardinality non-numeric features and used one-hot-encoder to convert categorical “interest-level” column into numeric for regression modeling. Used SelectKBest with f_regression score function to select best 15 features based on univariate linear regression test to find the correlation factor between each feature and target. Used Ridge regressor to regularize the coefficient values by adding bias through alpha parameter. This would filter out the noise in input data when modeling and lower the variance of predicted values. Used RidgeCV to apply k-fold cross validation to optimize the value of alpha, as an alternative way of sweeping the alpha value and plotting mse of validation data. The regularization helped to avoid overfitting. It lowered the training score in favor of better validation score.
Used libraries:
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
sklearn.model_selection.train_test_split
from sklearn.feature_selection import f_regression, SelectKBest
from sklearn.preprocessing import StandardScaler
seaborn
matplotlib.pyplot
from category_encoders import OneHotEncoder
https://github.com/skhabiri/PredictiveModeling-LinearModels-u2s1/tree/master/RidgeRegression-m3
