# https://www.kaggle.com/code/herdinsurya/youtube-views-prediction-machine-learning/notebook
# Dataset Source : https://www.kaggle.com/code/herdinsurya/youtube-views-prediction-machine-learning/data?select=CAvideos.csv

import warnings
warnings.filterwarnings("ignore")

import os

dataPath = r"C:\Users\gauta\PycharmProjects\MachineLearning\ML Projects\Videos_Views_Prediction\Dataset"
fileName = "CAvideos.csv"

filePath = os.path.join( dataPath, fileName )

import pandas as pd

####################################Step 1 - Load Data#################################
# Reading the csv file having videos data
data = pd.read_csv( filePath )

####################################Step 2 - Describe Data#############################
# Identifying shapes : rows - > 40881 and Cols -> 16 and datatypes of each columns
print( f"data.shape = { data.shape }\ndata.columns :-\n{ list( data.columns ) }\n" )
print( f"data.dtypes :- \n{ data.dtypes }\n" )


###########################Step 3 - Exploratory Data Analysis#################
# Seperating categorical and Numerical Columns


# n(categoricalCols) = 8 = ['video_id', 'trending_date', 'title', 'channel_title'
#                        , 'publish_time', 'tags', 'thumbnail_link', 'description']
categoricalCols = list( data.dtypes[ data.dtypes == 'object' ].index )
print( f"categoricalCols = { categoricalCols } and n(categoricalCols) = { len(categoricalCols) }\n" )



# n(NumericalCols) = 8 = ['category_id', 'views', 'likes', 'dislikes', 'comment_count'
#                       , 'comments_disabled', 'ratings_disabled', 'video_error_or_removed']
NumericalCols   = list( data.dtypes[ data.dtypes != 'object' ].index )
print( f"NumericalCols = { NumericalCols } and n(NumericalCols) = { len(NumericalCols) }\n" )

##########Step 3.1 - Function - Describing NumericalCols Columns for Skewness analysis
print( f"data[NumericalCols].describe() :- \n{data[NumericalCols].describe()}\n" )

def findSkewCols( data, NumericalCols ):
      skeCols = []
      
      dataDescribe = data[NumericalCols].describe()
      for i in dataDescribe:
             diff = abs( dataDescribe[i]['mean'] - dataDescribe[i]['50%'] )
             gapPercent = diff/dataDescribe[i]['mean']

             # print( f"i = { i } and gapPercent = { gapPercent }" )
             if( gapPercent > 0.50 ):
                   skeCols.append( i )

      return skeCols

##############Step 3.1.1 - Numerical Data Conclusion
# ['views', 'likes', 'dislikes', 'comment_count'] are Skews
skewCols = findSkewCols( data, NumericalCols )
print( f"skewCols = { skewCols }\n" )

###############Step 3.1.2 - Categorical Data Conclusion
for i in categoricalCols:
      print( f"Data in {i} has { len(data[i].unique()) } unique values" )

print()
"""
Data in video_id has 24427 unique values
Data in trending_date has 205 unique values
Data in title has 24573 unique values
Data in channel_title has 5076 unique values
Data in publish_time has 23613 unique values
Data in tags has 20157 unique values
Data in thumbnail_link has 24422 unique values
Data in description has 22346 unique values
"""

##########Step 3.2 - Graphical Approaching#########

##########Step 3.2.1 - Univariate Analysis#########
import matplotlib.pyplot as plt
import seaborn as sns

# Setting up Graph Styles
sns.set(rc = {'figure.figsize':( 2*len(NumericalCols), len(NumericalCols) ) } )
sns.set_style("whitegrid")
sns.color_palette("dark")
# plt.style.use("fivethirtyeight")

# Creating the Figure Size
plt.figure(figsize=( 2*len(NumericalCols), len(NumericalCols) ) )
for i in range(0, len(NumericalCols)):
    plt.subplot(1, len(NumericalCols), i+1)
    sns.boxplot(y=data[NumericalCols[i]],color='green',orient='v')
    plt.tight_layout()

# plt.show()
#plt.savefig('Outlier_boxplot.png')

# Conclusion :- 
# It can be seen in the boxplot graph above that the views, likes, dislikes, comment_count
# features have many outliers so that logarithmic transformations are needed for
# these features

# Question : Why logarithmic transformations ?
# Answer : 
#           Logarithmic transformations can do following things,

# 1. Data compression: When dealing with data that covers a large range of values
#  , logarithmic transformations can be used to compress the range of values.
#    This can make it easier to visualize and analyze the data, as well as reducing
#    the impact of extreme values on statistical analyses.

# 2. Linearization: In some cases, data that is non-linear can be transformed using a
#    logarithmic function to make it linear. This can make it easier to apply linear
#    models, such as regression analysis.

# 3. Symmetry: Some distributions of data can be skewed or asymmetric, which can make
#    it difficult to analyze using traditional statistical methods. By applying a
#    logarithmic transformation, it is possible to make the distribution more symmetric
#  , which can improve the accuracy and reliability of statistical tests.

# 4. Percentage changes: Logarithmic transformations can be used to represent percentage
#    changes. For example, if a variable is transformed by taking the natural logarithm,
#    then the difference between two values on the transformed scale represents the
#    percentage change in the original variable. Overall, logarithmic transformations
#    are a useful tool in many different areas of analysis, including statistics,
#    finance, and engineering. They can help to make complex data more manageable and
#    easier to work with, allowing analysts and researchers to draw more accurate and
#    meaningful conclusions from their data. 

##########Step 3.2.2 - Multivariate Analysis#########

# Finding Co-relation of Numerical Columns
corr_= data.loc[ : , NumericalCols ].corr()
plt.figure(figsize=(16,10))
sns.heatmap(corr_, annot=True, fmt = ".2f", cmap = "BuPu");
# plt.savefig('Corr_heatmap.png');
# plt.show()

# Conclusion :-
#           There are 3 features that have a strong positive correlation to views Columns
#           that is = likes, dislikes, and comment_count features


###########################Step 4 - Data Preparation#################

########## Step 4.1 - Identifying Null ValuesnullCols( data ) ##########
def nullCols( data ):
      data_missing_value = data.isnull().sum().reset_index()
      data_missing_value.columns = ['feature','missing_value']
      data_missing_value['percentage'] = round(
                              ( data_missing_value['missing_value']/len(data))*100,2
                                               )
      data_missing_value = data_missing_value.sort_values('percentage', ascending=False).reset_index(drop=True)
      data_missing_value = data_missing_value[data_missing_value['percentage']>0]

      # print( data_missing_value )
      return data_missing_value

data_missing_value = nullCols( data )
print( f"Before, Null Columns with Percentage :- \n{data_missing_value}\n" )

########## Step 4.2 - Drop Column/ Filling Missing Column with Mode ##########

# As the null values are very less so we are filling the values by mode
for i in data_missing_value['feature']:
      data[i].fillna( data[i].mode()[0], inplace = True )

data_missing_value = nullCols( data )
print( f"After, Null Columns with Percentage :- \n{data_missing_value}\n\n" )

########## Step 4.3 - Duplicate Values Fixing ###########
print( f"Before Duplicate, duplicate n(row) = { data.duplicated().sum() }\n" )
data.drop_duplicates( inplace = True )
print( f"After Duplicate, duplicate n(row) = { data.duplicated().sum() }\n" )

########## Step 4.4 - Outlier Fixing ###########
print( f"Before Duplicate, duplicate n(row) = { data.duplicated().sum() }\n" )
data.drop_duplicates( inplace = True )
print( f"After Duplicate, duplicate n(row) = { data.duplicated().sum() }\n" )

# As discussed above in 3.2.1 segmant that views, likes, dislikes, comment_count columns has
# some outliers

def detectOutlierColumns( data, exceptionalCols ):
      data.drop( exceptionalCols, inplace = True, axis = 1 )
      
      outlierColumns = []
      
      for col in data:
            colDescribe = data[col].describe()

            # print( col, colDescribe )
            if( ('25%' in colDescribe) and ('75%' in colDescribe) ):
            
                  iqr = 1.5 * ( colDescribe['75%'] - colDescribe['25%'] )

                  lowerOutlier = colDescribe['25%'] - iqr
                  upperOutlier = colDescribe['25%'] + iqr

                  # print( col, lowerOutlier, upperOutlier )

                  if( ( len( data['views'][ data['views']<lowerOutlier ] ) > 0 )
                        or ( len( data['views'][ data['views']>upperOutlier ] ) > 0 )
                  ):
                        outlierColumns.append( col )

      return outlierColumns

outlierColumns = detectOutlierColumns( data.loc[ :,NumericalCols], exceptionalCols = 'category_id' )
print( f"Outlier Columns = { outlierColumns }\n" )

# Applying logarithmic transformations on Outlier Columns
import numpy as np

print( f"Normal Outlier Columns :- \n{ data.loc[ :, outlierColumns ] }\n" )

df_pre = data.copy()
for col in outlierColumns:
    df_pre['log_'+col]= (data[col]+1).apply(np.log)

print( f"Log Transformation Columns :- \n{ df_pre.iloc[ :, data.shape[1] : ] }\n" )

from sklearn.preprocessing import MinMaxScaler

for col in outlierColumns:
    df_pre['std_'+col]= MinMaxScaler().fit_transform(df_pre[col].values.reshape(len(df_pre), 1))

print( f"MinMaxScaler Columns :- \n{ df_pre.iloc[ :, df_pre.shape[1] - len(outlierColumns ) : ] }\n" )

# Question : What Why MinMaxScaling ?
# Answer : ﻿MinMaxScaler is a data preprocessing technique commonly used in machine learning
# to scale numeric features in a dataset to a fixed range. Specifically, it scales each
# feature to a specified minimum and maximum value, typically between 0 and 1.

# The MinMaxScaler transformation is useful in many machine learning algorithms because
# it ensures that all features are on the same scale. Features on different scales can have
# different effects on the model, and so scaling can help the algorithm to better understand
# the importance of each feature.

# The MinMaxScaler formula for scaling a feature x is:
# x_scaled = (x - x_min) / (x_max - x_min) Where x_min is the minimum value of the feature,
# and x_max is the maximum value of the feature.

# In general, MinMaxScaler is a good choice when the distribution of the data is not
# Gaussian and when the range of the features varies widely. It is also useful when
# comparing features that have different units of measurement, as it scales them to a
# common range.

###########################Step 5 - Split Train & Test#################
sns.set(rc={'figure.figsize':( 10,len(outlierColumns ) )})
sns.heatmap(df_pre.iloc[ :, df_pre.shape[1] - len(outlierColumns ) : ].corr(), annot=True)
# plt.savefig('split train test.png')
# plt.show()

# Spliting data into x and y
x = df_pre.loc[ :, 'std_likes' : 'std_comment_count' ]
y = df_pre['std_views']


from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.2, random_state=0)

###########################Step 6 - Making Model Evaluation Model #################
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
def eval_regression(model, pred, xtrain, ytrain, xtest, ytest, modelName = ""):
    print(modelName, " : MAE: %.2f" % mean_absolute_error(ytest, pred)) # The MAE
    print(modelName, " : RMSE: %.2f" % mean_squared_error(ytest, pred, squared=False)) # The RMSE
    print(modelName, ' : R2 score: %.2f' % r2_score(ytest, pred)) # Explained variance score: 1 is perfect prediction

###########################Step 7 - Training and Model Fitting #################
maxAccuracy, selectedModel = 0, ""

####### 7.1 - Fit Simple Linear Regression Model #######

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(xtrain, ytrain)
pred = regressor.predict(xtest)
score = regressor.score(xtest,ytest)

if( score > maxAccuracy ):
      maxAccuracy = score
      selectedModel = "Simple Linear Regression Model"

print(f'LinearRegression : Coefficients: {regressor.coef_}') # The slope
print(f'LinearRegression : Intercept: {regressor.intercept_}') # The Intercept
eval_regression(regressor, pred, xtrain, ytrain, xtest,ytest, "LinearRegression")
print( f"LinearRegression Score = { score }\n" )

# Question : Why Linear Regression ?
# Answer : Linear regression is a type of supervised machine learning algorithm used to
# predict a continuous target variable based on one or more predictor variables.
# The basic idea behind linear regression is to model the relationship between the
# predictor variables and the target variable by fitting a straight line (or hyperplane,
# in the case of multiple predictors) to the data. This line represents the best linear
# approximation of the true relationship between the variables. Once the line is fitted,
# it can be used to make predictions for new data.

####### 7.2 - Fit Ridge Regularization Model #######
from sklearn.linear_model import Ridge
ridge_model = Ridge()
ridge_model.fit(xtrain, ytrain)
pred = ridge_model.predict(xtest)
score = ridge_model.score(xtest,ytest)

if( score > maxAccuracy ):
      maxAccuracy = score
      selectedModel = "Ridge Regularization Model"

print(f'Ridge Regularization Model : Coefficients: {ridge_model.coef_}') # The slope
print(f'Ridge Regularization Model : Intercept: {ridge_model.intercept_}') # The Intercept
eval_regression( ridge_model, pred, xtrain, ytrain, xtest, ytest, "Ridge Regularization Model")
print( f"Ridge Regularization Model Score = { score }\n" )

# Question : Why Ridge Regularization Model ?
# Answer : Ridge regularization is a technique used in linear regression to prevent
# overfitting and improve the accuracy and generalization performance of the model.
# It achieves this by adding a penalty term to the linear regression objective function,
# which shrinks the coefficients towards zero, effectively reducing the model complexity.

# The penalty term in ridge regularization is proportional to the square of the L2 norm
# of the coefficients, and is controlled by a hyperparameter called the regularization
# parameter. Ridge regularization is particularly useful in situations where the number
# of predictor variables is large and the correlation between them is high, which can cause
# instability in the model and lead to overfitting. By penalizing large coefficients, ridge
# regression can reduce the variance of the model, which can improve the accuracy and
# generalization performance.

####### 7.3 - Randomized Search + Ridge Regularization Model #######
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

alpha = [200, 230, 250,265, 270, 275, 290, 300, 500] # alpha
hyperparameters = dict(alpha=alpha)

ridge_model = Ridge()
clf = RandomizedSearchCV(ridge_model, hyperparameters, cv=5, random_state=0, scoring='r2')
best_model = clf.fit(xtrain, ytrain)
pred = best_model.predict(xtest)
score = best_model.score(xtest,ytest)

if( score > maxAccuracy ):
      maxAccuracy = score
      selectedModel = "Randomized Search + Ridge Regularization Model"

eval_regression( best_model, pred, xtrain, ytrain, xtest, ytest, "Randomized Search + Ridge Regularization Model")
print( f"Randomized Search + Ridge Regularization Model Score = { score }\n" )

####### 7.4 - Fit Lasso Regularization Model #######
from sklearn.linear_model import Lasso
lasso_model = Lasso()
lasso_model.fit(xtrain, ytrain)
pred = lasso_model.predict(xtest)
score = lasso_model.score(xtest,ytest)

if( score > maxAccuracy ):
      maxAccuracy = score
      selectedModel = "Lasso Regularization + Ridge Regularization Model"

eval_regression( best_model, pred, xtrain, ytrain, xtest, ytest, "Lasso Regularization")
print( f"Lasso Regularization + Ridge Regularization Model Score = { score }\n" )

# Question : Why Lasso Regularization Model ?
# Answer : Lasso (Least Absolute Shrinkage and Selection Operator) regularization is a
# technique used in linear regression to prevent overfitting and improve the accuracy and
# interpretability of the model.
# Lasso regularization is particularly useful in situations where there are many predictor
# variables, and many of them may be irrelevant or redundant.

# Basically, lasso regularization is a useful technique in linear regression that can help improve the
# performance and interpretability of the model, especially when dealing with high
# dimensional data and the need for variable selection.

####### 7.5 - Randomized Search + Fit Lasso Regularization Model #######
alpha = [0.02, 0.024, 0.025, 0.026, 0.03] # alpha or lambda
hyperparameters = dict(alpha=alpha)

lasso_model = Lasso()
clf = RandomizedSearchCV(lasso_model, hyperparameters, cv=5, random_state=0, scoring='r2')
best_model = clf.fit(xtrain, ytrain)
pred = best_model.predict(xtest)
score = best_model.score(xtest,ytest)

if( score > maxAccuracy ):
      maxAccuracy = score
      selectedModel = "Randomized Search + Lasso Regularization Model"

eval_regression( best_model, pred, xtrain, ytrain, xtest, ytest, "Randomized Search + Lasso Regularization Model")
print( f"Randomized Search + Lasso Regularization Model Score = { score }\n" )

####### 7.6 - Fit Elastic Net Regularization Model #######
from sklearn.linear_model import ElasticNet
elasticnet_model = ElasticNet()
elasticnet_model.fit(xtrain, ytrain)
pred = elasticnet_model.predict(xtest)
score = elasticnet_model.score(xtest,ytest)

if( score > maxAccuracy ):
      maxAccuracy = score
      selectedModel = "Elastic Net Regularization"

eval_regression( elasticnet_model, pred, xtrain, ytrain, xtest, ytest, "Elastic Net Regularization")
print( f"Elastic Net Regularization Score = { score }\n" )

# Question : Why Elastic Net Regularization Model ?
# Answer : Elastic Net regularization is a technique used in linear regression to prevent
# overfitting and improve the accuracy and stability of the model. It is a combination of
# Ridge (L2) and Lasso (L1) regularization, and aims to leverage the benefits of both
# techniques while avoiding their limitations.
# Basically, elastic net regularization is a useful technique in linear regression that can
# help improve the performance and stability of the model, especially when dealing with
# high-dimensional data, the need for variable selection, and the presence of collinearity
# among the predictors.

####### 7.7 - Randomized Search + Fit Elastic Net Regularization Model #######
alpha = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.0, 1.0, 10.0, 100.0]
l1_ratio = np.arange(0, 1, 0.01)

from sklearn.linear_model import ElasticNet
elasticnet_model = ElasticNet()
clf = RandomizedSearchCV(elasticnet_model, hyperparameters, cv=5, random_state=0, scoring='r2')
best_model = clf.fit(xtrain, ytrain)
pred = best_model.predict(xtest)
score = best_model.score(xtest,ytest)

if( score > maxAccuracy ):
      maxAccuracy = score
      selectedModel = "Randomized Search + Elastic Net Regularization Model"

eval_regression( best_model, pred, xtrain, ytrain, xtest, ytest, "Randomized Search + Elastic Net Regularization Model")
print( f"Randomized Search + Elastic Net Regularization Model Score = { score }\n" )

####### 7.8 - Fit Decision Tree Model #######
from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor()
dt.fit(xtrain, ytrain)
pred = dt.predict(xtest)
score = dt.score(xtest,ytest)

if( score > maxAccuracy ):
      maxAccuracy = score
      selectedModel = "Decision Tree Model"

eval_regression( dt, pred, xtrain, ytrain, xtest, ytest, "Decision Tree Model")
print( f"Decision Tree Model Score = { score }\n" )

# Question : Why Decision Tree Model ?
# Answer : A decision tree model works by recursively splitting the dataset into smaller
# subsets based on the values of one or more input features, until the subsets are as
# homogeneous as possible in terms of the target variable.

# Basically, decision trees are a popular and useful technique in machine learning,
# particularly when the goal is to achieve high interpretability or when dealing with both
# ategorical and numerical data. They can be particularly useful for feature selection and
# can be improved by combining them with other techniques to address their limitations.

####### 7.9 - Fit Random Forest Model #######
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(random_state=0)
best_model1 = rf.fit(xtrain, ytrain)
pred = rf.predict(xtest)
score = best_model1.score(xtest,ytest)

if( score > maxAccuracy ):
      maxAccuracy = score
      selectedModel = "Random Forest Model"

eval_regression(rf, pred, xtrain, ytrain, xtest, ytest, "Random Forest Model")
print( f"Random Forest Model Score = { score }\n" )

# Question : Why Random Forest Model ?
# Answer : Random Forest works by building multiple decision trees, each trained on a
# randomly sampled subset of the data and features, and then aggregating the predictions
# of the individual trees to produce the final output.

# Basically, Random Forest is a powerful and widely used technique in machine learning,
# particularly when the goal is to achieve high accuracy and robustness.It can be
# particularly useful for complex datasets and can provide insights into the most important
# features for the task at hand. However, it may be less suitable when the goal is to
# achieve high interpretability or when computational efficiency is a priority.

####### 7.10 - Fit Support Vector Regressor Model #######
from sklearn.svm import SVR

svr = SVR()
svr.fit(xtrain, ytrain)
pred = svr.predict(xtest)
score = svr.score(xtest,ytest)

if( score > maxAccuracy ):
      maxAccuracy = score
      selectedModel = "SVR Model"

eval_regression(svr, pred, xtrain, ytrain, xtest, ytest, "SVR Model")
print( f"SVR Model Score = { score }\n" )

print( f"{selectedModel} has shown Maximum Accuracy = { round(maxAccuracy,2)*100 }%" )

# Question : Why SVR(Support Vector Regressor) ?
# Answer : SVR(Support Vector Regressor) works by finding a hyperplane that maximally fits
# the data points within a given margin, while also minimizing the prediction error.
# SVR has several advantages over other regression techniques.

# Firstly, it is effective in dealing with high-dimensional data and large datasets, and
# can handle both continuous and categorical data.

# Secondly, it is robust to outliers and noise, making it a useful technique for real-world
# datasets.

# Thirdly, it can be used with different types of kernel functions, such as linear,
# polynomial, and radial basis function (RBF) kernels, allowing for greater flexibility
# in modeling complex relationships between the input and output variables

# Basically, SVR is a powerful and widely used technique in machine learning, particularly
# for regression tasks where the goal is to achieve high accuracy and robustness. It can
# handle complex datasets and provides a range of options for modeling different types of
# relationships between the input and output variables. However, it may be less suitable
# when computational efficiency or interpretability is a priority.

###########################Step 8 - Conclusive Accuracies###########################
# Random Forest Model Score = 0.8389816171508484
# Decision Tree Model Score = 0.7301737795195783
# LinearRegression Score = 0.7204449761669244
# Ridge Regularization Model Score = 0.7146482191466955
# Randomized Search + Ridge Regularization Model Score = 0.1772877415801286
# Lasso Regularization + Ridge Regularization Model Score = -0.00024163497356921582
# Randomized Search + Lasso Regularization Model Score = -0.00024163497356921582
# Elastic Net Regularization Score = -0.00024163497356921582
# Randomized Search + Elastic Net Regularization Model Score = -0.00024163497356921582
# SVR Model Score = -12.649190519722103
