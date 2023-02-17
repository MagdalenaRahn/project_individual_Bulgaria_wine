# imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression, RFE




# SelectKBest function

# df = dataframe,
# scaled_cols = columns to scale, entered as a list, i.e, ['variable_name'],
# target_var = target variable, entered as a string, i.e, 'variable_name'
# kk = the k number of features to select / return

def select_best(df, scaled_cols, target_var, kk):
 
    '''
    This function takes in the predictors (X), the target (y) and 
    the number of features to select (k) and returns the names of
    the top k selected features based on the SelectKBest class. 
    '''
    
    # creating a copy of original df in order to avoid messes and permanently scaled data later
    df_copy = df.copy()
    
    # scaling the data in specific columns, reassigning the scaled numbers to the column names
    mms = MinMaxScaler()
    df_copy[scaled_cols] = mms.fit_transform(df_copy[scaled_cols])

    # feature array / df with continuous features (X, ...) and target (y)
    X_train_scaled = df_copy[scaled_cols]
    y_train = df_copy[target_var]

    # create an instance of the SelectKBest object
    selector = SelectKBest(f_regression, k = kk)

    # fit the object to data : .fit(features, target variable)
    selector.fit(X_train_scaled, y_train)
    
    # masking the values by assigning a variable to the T/F in order to apply it to the columns
    selector_rankings = selector.get_support()

    # see only the column names relevant to our analysis
    best = X_train_scaled.columns[selector_rankings]
    
    return best



# function to turn income bins into integers

def to_integer(coup):
    
    '''
    this function intakes the original dataframe and converts alreay-
    dummied columns (income, weather, bar) into integers.
    '''

    coup['income_100000_or_more'] = coup['income_100000_or_more'].astype('int')
    coup['income_12500_24999'] = coup['income_12500_24999'].astype('int')
    coup['income_25000_37499'] = coup['income_25000_37499'].astype('int')
    coup['income_37500_49999'] = coup['income_37500_49999'].astype('int')
    coup['income_50000_62499'] = coup['income_50000_62499'].astype('int')
    coup['income_62500_74999'] = coup['income_62500_74999'].astype('int')
    coup['income_75000_87499'] = coup['income_75000_87499'].astype('int')
    coup['income_less_than_12500'] = coup['income_less_than_12500'].astype('int')
    coup['income_87500_99999'] = coup['income_87500_99999'].astype('int')
    
    coup['weather_Sunny'] = coup['weather_Sunny'].astype('int')
    coup['weather_Rainy'] = coup['weather_Rainy'].astype('int')
    coup['weather_Snowy'] = coup['weather_Snowy'].astype('int')
    
    coup['bar_1_3'] = coup['bar_1_3'].astype('int')
    coup['bar_4_8'] = coup['bar_4_8'].astype('int')
    coup['bar_never'] = coup['bar_never'].astype('int')
    coup['bar_less1'] = coup['bar_less1'].astype('int')
    coup['bar_gt8'] = coup['bar_gt8'].astype('int')

    return coup





# RFE function

# df = dataframe,
# target_var = target variable (y-value), the column to drop, entered as a string, i.e, 'variable_name'
# fts = n_features_to_select
# dummy_columns = columns of which to make dummies, entered as a list, i.e ['variable_name']

def rfe_function(df, fts, target_var, dummy_columns):
    
    '''This function takes in the predictors, the target and the 
    number of features to select and returns the top k-features 
    based on the RFE class.
    '''
    
    # dropping 'target_var' to allow for fair evaluation of data
    X_train = df.drop(columns = target_var)

    # make dummies of categorical columns, then reassign to variable X_train
    # X_train = pd.get_dummies(X_train, columns = dummy_columns) 
                # ^ this line is for when dummies not already made
    X_train = df[dummy_columns]
    y_train = df[target_var]

    # RFE uses a machine learning model (here, linear regression) 
    #   to determine the 2 features with the most predictive capability
    rfe = RFE(LinearRegression(), n_features_to_select = fts)

    # fitting 
    rfe.fit(X_train, y_train)

    # ranking of the categorical columns and aligning them with their column names
    ranking = rfe.ranking_
    features = X_train.columns.tolist()

    # turning ranks & columns into a df
    df = pd.DataFrame({'ranking' : ranking,
                       'feature' : features})

    df = df.sort_values('ranking')
    
    return df