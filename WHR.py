##Recordings are in the README

import pandas as pd
import numpy as np
import os 
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
%matplotlib inline
pd.options.display.float_format = '{:.2f}'.format

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import GradientBoostingClassifier

#Get a look at the data and missing values
filename = os.path.join(os.getcwd(), "data", "WHR2018Chapter2OnlineData.csv")
df = pd.read_csv(filename, header=0)

df_summ = df.describe()
df_summ

df.isnull().values.any()

df.isnull().head()

nan_count = np.sum(df.isnull(), axis = 0)
nan_count

condition = nan_count != 0 # look for all columns with missing values

col_names = nan_count[condition].index # get the column names
print(col_names)

nan_cols = list(col_names) # convert column names to list
print(nan_cols)

nan_col_types = df[nan_cols].dtypes
nan_col_types

df = df[df.year.isin(range(2015,2016))]
df.drop(columns = ['country','year','Positive affect','Negative affect','Delivery Quality','GINI index (World Bank estimate)','GINI index (World Bank estimate), average 2000-15','gini of household income reported in Gallup, by wp5-year','Standard deviation/Mean of ladder by country-year'],inplace=True)
'''
Cleaning of infinity and NaN Values: After using methods such as winstorization and fillna to detect outliers and replace missing values,
I ensured there were no NaN or infinity values by looking for the location of remaining missing values or using the np.inf function to detect any infinity values. 
I got 0 and false as a result of checking for both instances, which led to having to use a function to exclude any rows that contains those values. I am not sure why after fitting, I kept getting the error of there being a NaN or infinity value. 
However, since for a majority of my data, missing data was filled and outliers were replaced,I didn't mind making the decision of dropping 1-2 rows to get my model to fit.
'''
df =df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]
​
df =df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]
​
'''
Detecting any outliers and replacing them, after finding and replacing any missing data, two columns have missing values or too high of a value for the model to be fitted
therefore I'm going to add a column with the winsorized Version of the original column for scoial support and healthy life expectancy
'''
df['social_support_win'] = stats.mstats.winsorize(df['Social support'], limits=[0.01, 0.01])

df['health_expect_win'] = stats.mstats.winsorize(df['Healthy life expectancy at birth'], limits=[0.01, 0.01])

df.head(15)

'''
Detect and Replace Missing Values: Get the mean of a designated column and assign to a variable. Replace any missing values of relevant columns using the mean taken in previous row.
'''
log_gdp = df['Log GDP per capita'].mean()
df['Log GDP per capita'].fillna(value = log_gdp,inplace=True)
perception = df['Perceptions of corruption'].mean()
df['Perceptions of corruption'].fillna(value = perception,inplace=True)
generosity = df['Generosity'].mean()
df['Generosity'].fillna(value = generosity,inplace=True)
democratic = df['Democratic Quality'].mean()
df['Democratic Quality'].fillna(value =democratic,inplace=True)
confidence = df['Confidence in national government'].mean()
df['Confidence in national government'].fillna(value = confidence,inplace=True)
freedom = df['Freedom to make life choices'].mean()
df['Freedom to make life choices'].fillna(value = freedom,inplace=True)
social= df['Social support'].mean()
df['Social support'].fillna(value = social,inplace=True)
health = df['Healthy life expectancy at birth'].mean()
df['Healthy life expectancy at birth'].fillna(value =health,inplace=True)
​
df.head(50)
​
#Use the np.sum to see if there's any missing values in each column
nan_count = np.sum(df.isnull(), axis=0)
print(nan_count)

df.drop(columns = ['social_support_win','health_expect_win'],axis=1)
#make a list of the relevant features and create label+features
features = ['Log GDP per capita','Social support','Perceptions of corruption','Confidence in national government', 'Standard deviation of ladder by country-year', 'Freedom to make life choices', 'Generosity', 'Democratic Quality','Healthy life expectancy at birth']

y = df['Life Ladder']
X = df.drop(columns = 'Life Ladder',axis=1)

print(X)

#Going to use the train test split function and also test with a random_state of 42 and test around with test_size from .15-.35 in intervals of .5.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)

# Create theLinearRegression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test data 
prediction = model.predict(X_test)
prediction


print('Model Summary:\n')

# Print intercept (alpha)
print('Intercept:')
print('intercept = ' , model.intercept_)

# Print weights
print('\nWeights:')
i = 0
for w in model.coef_:
    print('w_',i+1,'= ', w, ' [ weight of ', features[i],']')
    i += 1


#Save and print root mean squared error + r squared
lr_rmse = mean_squared_error(y_test,prediction,squared=False)
lr_r2 = r2_score(y_test,prediction)

print('\nModel Performance\n\nRMSE =   %.2f'
      % np.sqrt(mean_squared_error(y_test, prediction)))
# Closer to  1 is preferrable
print(' R^2 =   %.2f'
      % r2_score(y_test, prediction))


#Creating Gradient Boosting Decision Tree
print('Begin GBDT Implementation...')
​
# 1. Create the  model object below and assign to variable 'gbdt_model'
gbdt_model = GradientBoostingRegressor(max_depth=2,n_estimators=240)
​
# 2. Fit the model to the training data below
# YOUR CODE HERE
gbdt_model.fit(X_train,y_train)
​
print('End')

#Make Predictions on the Test Dataset and Compute the RMSE and R2

# 1. Use predict() to use the fitted model to make predictions on the test dataset
y_gbdt_pred = gbdt_model.predict(X_test)
​
# 2. Compute the RMSE using mean_squared_error() 
gbdt_rmse = mean_squared_error(y_test,y_gbdt_pred,squared=False)
​
# 3. Compute the R2 score using r2_score()
gbdt_r2 = r2_score(y_test,y_gbdt_pred)
           
#format the RMSE and R2
print('[GBDT] Root Mean Squared Error: {0}'.format(gbdt_rmse))
print('[GBDT] R2: {0}'.format(gbdt_r2))     

#Create Random Forest Model
print('Begin RF Implementation...')

# 1. Create the  model object below and assign to variable 'rf_model'
rf_model = RandomForestRegressor(max_depth=17,n_estimators = 500)

# 2. Fit the model to the training data below
rf_model.fit(X_train,y_train)


print('End')


# 1. Use the fitted model to make predictions on the test data
y_rf_pred = rf_model.predict(X_test)

# 2. Compute the RMSE using mean_squared_error()
rf_rmse = mean_squared_error(y_rf_pred,y_test,squared=False)

# 3. Compute the R2 score using r2_score()
rf_r2 = r2_score(y_test,y_rf_pred)

                   
print('[RF] Root Mean Squared Error: {0}'.format(rf_rmse))
print('[RF] R2: {0}'.format(rf_r2))


#Visualization
RMSE_results = [lr_rmse,gbdt_rmse,rf_rmse]
R2_Results = [lr_r2,gbdt_r2,rf_r2]
labels = ['LR', 'GBDT', 'RF']

rg = np.arange(3)
width = 0.35
plt.bar(rg, RMSE_Results, width, label="RMSE")
plt.bar(rg + width, R2_Results, width, label='R2')
plt.xticks(rg + width/2, labels)
plt.xlabel("Models")
plt.ylabel("RMSE/R2")
plt.ylim([0, 1])

plt.title('Model Performance')
plt.legend(loc='upper left', ncol=2)
plt.show()


​
​
