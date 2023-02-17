import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error

from sklearn.cluster import KMeans
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import TweedieRegressor
from math import sqrt
from scipy.stats import pearsonr, spearmanr
from scipy import stats

from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier


# turn off pink boxes for demo
import warnings
warnings.filterwarnings("ignore")

seed = 23



# heatmap 

def heat(train):
    
    '''
    this function takes in a dataframe and makes a heatmap
    '''
    
    plt.figure(figsize = (16,8))
    sns.heatmap(train.corr(), cmap = 'Oranges', cbar = True,
                annot_kws = {'fontsize' : 8}, mask = np.triu(train.corr()))

    plt.show()
    
    


# function to plot income in initial exploration : monovariable

def plot_income(coup):
    
    '''
    this function takes in a dataframe and plots
    income distribution
    '''
    
    result = coup.groupby(["income"])['Y'].median().reset_index().sort_values('income', ascending = False)

    fig = sns.barplot(x = 'income', y = "Y", data = coup, order = result['income'])

    plt.legend([],[])

    fig.set(xlabel = 'Income Group In USD', ylabel = 'Liklihood Of Coupon Acceptance')

    plt.xticks(np.arange(-1, 8, step = 0.2))

    labels = ['Less than $12500', '$12500 - $24999', '$25000 - $37499','$37500 - $49999',
              '$50000 - $62499', '$62500 - $74999', '$75000 - $87499', '$87500 - $99999',
              '$100000 or More']
    
    plt.xticks(ticks = (-1,0,1,2,3,4,5,6,7), labels = labels, rotation = 45)
    
    plt.title('Income Groups')

    plt.show()



def tts_xy(train, val, test, target):
    
    '''
    This function splits train, val, test into X_train, X_val, X_test
    (the dataframe of features, exludes the target variable) 
    and y-train (target variable), etc
    '''

    X_train = train.drop(columns = [target])
    y_train = train[target]


    X_val = val.drop(columns = [target])
    y_val = val[target]


    X_test = test.drop(columns = [target])
    y_test = test[target]
    
    y_train = pd.DataFrame(y_train)
    y_val = pd.DataFrame(y_val)
    y_test = pd.DataFrame(y_test)
    
    return X_train, y_train, X_val, y_val, X_test, y_test



# function to create weather-cluster column

def weather3_col(df):
    
    '''
    this function will create a column on the scaled dataset 
    to allow for regression modelling
    '''
    
    df2 = df[['weather_Sunny', 'weather_Rainy', 'weather_Snowy']]
    
    kmeans = KMeans(n_clusters = 3, random_state = 23)

    kmeans.fit(df2)

    kmeans.predict(df2)
    
    df['weather3'] = kmeans.predict(df2)
    
    return df



# function to create bar frequency-cluster column

def bar_freq_col(df):
    
    '''
    this function will create a column on the scaled dataset 
    to allow for regression modelling
    '''
    
    df2 = df[['bar_1_3', 'bar_4_8', 'bar_gt8', 'bar_less1', 'bar_never']]
    
    kmeans = KMeans(n_clusters = 3, random_state = 23)

    kmeans.fit(df2)

    kmeans.predict(df2)
    
    df['bar_freq'] = kmeans.predict(df2)
    
    return df


# function to create income frequency-cluster column

def income_col(df):
    
    '''
    this function will create a column on the scaled dataset 
    to allow for regression modelling
    '''
    
    df2 = df[['income_100000_or_more',
              'income_12500_24999','income_25000_37499', 
              'income_37500_49999', 'income_50000_62499', 
              'income_62500_74999', 'income_75000_87499', 
              'income_87500_99999','income_less_than_12500']]
    
    kmeans = KMeans(n_clusters = 3, random_state = 23)

    kmeans.fit(df2)

    kmeans.predict(df2)
    
    df['income_col'] = kmeans.predict(df2)
    
    return df


# t-test function

def t_test(a, b):
    '''
    This function will take in two arguments in the form of a continuous and discrete 
    variable and runs an independent t-test and prints whether or not to reject the
    Null Hypothesis based on those results.
    '''
    alpha = 0.05
    t, p = stats.ttest_ind(a, b, equal_var = False)
    print("T-statistic is : ", t)
    print("")
    print("P-value is : ", p/2)
    print("")
    if p / 2 > alpha:
        print("We fail to reject the null hypothesis.")
    elif t < 0 :
        print("We fail to reject the null hypothesis.")
    else:
        print(f"We reject the null hypothesis ; there is relationship between the target variable and the feature examined.")
        
        
        
# chi-square function

def chi_sq(a, b):
    '''
    This function will take in two arguments in the form of two discrete variables 
    and runs a chi^2 test to determine if the the two variables are independent of 
    each other and prints the results based on the findings.
    '''
    alpha = 0.05
    
    result = pd.crosstab(a, b)

    chi2, p, degf, expected = stats.chi2_contingency(result)

    print(f'Chi-square  : {chi2:.4f}') 
    print("")
    print(f'P-value : {p:.4f}')
    print("")
    if p / 2 > alpha:
        print("We fail to reject the null hypothesis.")
    else:
        print(f'We reject the null hypothesis ; there is a relationship between the target variable and the feature examined.')
        

        
# plotting coupon acceptance in different weather        
        
def plot_weather_coupon(train):
         
    '''
    this function plots coupon acceptance based on different
    types of weather. It takes in train dataset. Preferred
    to 'def plot_weather(train):' below.
    '''
    
    g = sns.diverging_palette(300, 10, s = 90)
    
    sns.countplot(train['weather3'], hue = train['Y'], palette = 'GnBu', orient = "h")
    plt.ylabel('Likelihood Of Coupon Acceptance')
    plt.xlabel('Weather Types')
   
    plt.title('Likelihood Of Coupon Acceptance In Different Weather')
    labels = ['Snowy', 'Sunny', 'Rainy']
    
    plt.legend(title='Coupon Acceptance', loc = 'upper right', labels=['No', 'Yes'])    

    plt.xticks(ticks = (0, 1, 2), labels = labels)
    plt.show()        
  
    
    

# plotting coupon acceptance for opposite direction   

def plot_opp_dir(train):
 
    '''
    this function plots coupon acceptance based on train data for
    people heading in the opposite direction
    '''
    
    g = sns.diverging_palette(300, 10, s = 90)
    
    sns.countplot(train['dir_opp'], hue = train['Y'], palette = 'GnBu', orient = "h")
    
    plt.ylabel('Likelihood Of Coupon Acceptance')
    plt.xlabel('Heading In The Opposite Direction ?')
   
    plt.title('Likelihood Of Coupon Acceptance When Heading In The Opposite Direction')
    
    labels = ['No', 'Yes']
    
    plt.legend(title = 'Coupon Acceptance', loc = 'upper left', labels = ['No', 'Yes'])    
    
    plt.xticks(ticks = (0, 1), labels = labels)
    plt.show()    

    

    
# plotting coupon acceptance based on bar visits

def plot_explore_bar(train):
    
    '''    
    this function takes in the train dataframe and plots
    frequency of bar visits per month against likelihood 
    of accepting the proposed coupon.
    '''
    
    g = sns.diverging_palette(300, 10, s = 10)
    
    sns.countplot(train['bar_freq'], hue = train['Y'], palette = 'GnBu', orient = "h")
    
    plt.ylabel('Likelihood Of Coupon Acceptance')
    plt.xlabel('Bar Visits Per Month')
   
    plt.title('Likelihood Of Coupon Acceptance Based Number Of Bar Visits Per Month')
    
    labels = ['Never', 'Less than 1', '1 to 3','4 to 8','More than 4']
    
    plt.legend(title = 'Coupon Acceptance', loc = 'upper right', labels = ['No', 'Yes'])    
    
    plt.xticks(ticks = (0,1,2,3,4), labels = labels, rotation = 45)
    plt.show()    
    
    
    
# plotting income groups and coupon acceptance
def plot_explore_inc(train):
    
    '''    
    this function takes in the train dataframe and plots
    income distribution against likelihood of accepting
    the proposed coupon.
    '''
    
    g = sns.diverging_palette(300, 10, s = 90)
    
    sns.countplot(train['income_col'], hue = train['Y'], palette = 'GnBu', orient = "h")
    
    plt.ylabel('Likelihood Of Coupon Acceptance')
    plt.xlabel('Income Group')
   
    plt.title('Likelihood Of Coupon Acceptance Based On Income Group')
    
    labels = ['Less than $12500', '$12500 - $24999', '$25000 - $37499','$37500 - $49999',
              '$50000 - $62499', '$62500 - $74999', '$75000 - $87499', '$87500 - $99999',
              '$100000 or More']
    
    plt.legend(title = 'Coupon Acceptance', loc = 'upper right', labels = ['No', 'Yes'])    
    
    plt.xticks(ticks = (-1,0,1,2,3,4,5,6,7), labels = labels, rotation = 45)
    plt.show()    
    
    

    
    
    
    
def mean_rmse(y_train, y_val, y_test):

    y_train = pd.DataFrame(y_train)
    y_val = pd.DataFrame(y_val)
    y_test = pd.DataFrame(y_test)

    coupon_pred_mean = y_train['Y'].mean()

    y_train['coupon_pred_mean'] = coupon_pred_mean
    y_val['coupon_pred_mean'] = coupon_pred_mean
    y_test['coupon_pred_mean'] = coupon_pred_mean

    rmse_train_mean = mean_squared_error(y_train['Y'], y_train['coupon_pred_mean']) ** 0.5
    rmse_val_mean = mean_squared_error(y_val['Y'], y_val['coupon_pred_mean']) ** 0.5
    rmse_test_mean = mean_squared_error(y_test['Y'], y_test['coupon_pred_mean']) ** 0.5

    return rmse_train_mean, rmse_val_mean, rmse_test_mean
    
    
    
    
def coupon_ols(df, col):
    
    '''
    this function runs the OLS Linear Regression model for the feature on an 
    entered dataframe with the feature 'column_name', comparing it to 
    accepting the coupon. It returns the RMSE baseline and the OLS RMSE.
    '''

    # rounding and setting target variable-mean name
    baseline_preds = round(df['Y'].mean(), 3)

    # create a dataframe
    predictions_df = df[[col, 'Y']]

    # MAKE NEW COLUMN ON DF FOR BASELINE PREDICTIONS
    predictions_df['baseline_preds'] = baseline_preds

    # our linear regression model
    ols_model = LinearRegression()
    ols_model.fit(df[[col]], df[['Y']])

    # model predictions from above line of codes with 'yhat' as variable name and append it on to df
    predictions_df['yhat'] = ols_model.predict(df[[col]])

    # computing residual of baseline predictions
    predictions_df['baseline_residual'] = predictions_df['Y'] - predictions_df['baseline_preds']

    # looking at difference between yhat predictions and actual preds ['Y']
    predictions_df['yhat_res'] = predictions_df['yhat'] - predictions_df['Y']

    # finding the RMSE in one step (x = original, y = prediction)
    OLS_preds_rmse = sqrt(mean_squared_error(predictions_df['Y'], predictions_df['baseline_preds']))
    print(f'The RMSE on the baseline against accepting the coupon is {round(OLS_preds_rmse,4)}.')

    # RMSE of linear regression model
    OLS_rmse = mean_squared_error(predictions_df['yhat'], predictions_df['Y'], squared = False)
    print(f'The RMSE for the OLS Linear Regression model was {round(OLS_rmse, 4)}.')
    
    return OLS_preds_rmse, OLS_rmse
    
    
    
    
    
# function for tweedie regresor    
 

def coupon_tweed(df, X_df, y_df, feature, target):
        
    '''
    This function intakes a scaled dataframe, and its X_ and y_ dataframes. 
    It also takes in the feature variable and compares it against
    the target variable.
    It returns the Tweedie Regressor RMSE and the baseline RMSE.
    '''

    # baseline on mean
    baseline_pred_t = round(df[target].mean(), 3)
    
    tweedie = TweedieRegressor()

    # fit the created object to training dataset
    tweedie.fit(X_df[[feature]], y_df)
    predictions_df = df[[feature, target]]

    # then predict on X_train
    predictions_df['tweedie'] = tweedie.predict(X_df[[feature]])
    predictions_df['baseline_pred_tweedie'] = baseline_pred_t


    # check the error against the baseline
    tweedie_preds_rmse = sqrt(mean_squared_error(predictions_df[target], predictions_df['baseline_pred_tweedie']))
    print(f'The RMSE on the baseline of weather against coupon acceptance is {round(tweedie_preds_rmse, 4)}.')

    # finding the error cf the baseline
    tweedie_rmse = sqrt(mean_squared_error(predictions_df[target], predictions_df['tweedie']))
    print(f'The RMSE for the Tweedie Regressor model was {round(tweedie_rmse,4)}.')

    return tweedie_preds_rmse, tweedie_rmse



# df for bar RMSE results

def bar_rmse_df(y_val, rmse_train_mean, OLS_preds_rmse, OLS_rmse, tweedie_preds_rmse, tweedie_rmse):
    
    '''
    this function returns a df with the RMSE and 
    baseline for the bar features
    '''
    
    # resetting y_val index 
    y_val.index.sort_values()
    y_val.reset_index(drop = True, inplace = True)

    # creating test results df
    bar_df = pd.DataFrame({'actual_y_val' : round(y_val, 4),
                           'baseline' : round(rmse_train_mean, 4),
                           'OLS_preds_RMSE' : round(OLS_preds_rmse, 4),
                           'OLS_RMSE' : round(OLS_rmse, 4),
                           'Tweedie_preds_RMSE':round(tweedie_preds_rmse, 4),
                           'Tweedie_RMSE' : round(tweedie_rmse, 4)})
    
    return bar_df



# df for weather RMSE results

def weather_rmse_df(y_val, rmse_train_mean, OLS_preds_rmse, OLS_rmse, tweedie_preds_rmse, tweedie_rmse):
    
    '''
    this function returns a df with the RMSE and baseline
    for the weather features
    '''

    # resetting y_val index 
    y_val.index.sort_values()
    y_val.reset_index(drop = True, inplace = True)

    # creating test results df
    weather_df = pd.DataFrame({'actual_y_val' : round(y_val, 4),
                                    'baseline' : round(rmse_train_mean, 4),
                                    'OLS_preds_RMSE' : round(OLS_preds_rmse, 4),
                                    'OLS_RMSE' : round(OLS_rmse, 4),
                                    'Tweedie_preds_RMSE':round(tweedie_preds_rmse, 4),
                                    'Tweedie_RMSE' : round(tweedie_rmse, 4)})
    
    return weather_df    






# df for weather RMSE results

def weather_rmse_val_df(y_df, rmse_val_mean, OLS_preds_rmse, OLS_rmse, tweedie_preds_rmse, tweedie_rmse):
    
    '''
    this function returns a df with the RMSE and baseline
    for the weather features
    '''

    # resetting y_val index 
    y_df.index.sort_values()
    y_df.reset_index(drop = True, inplace = True)

    # creating test results df
    weather_val_df = pd.DataFrame({'actual_y' : round(y_df, 4),
                                    'baseline' : round(rmse_val_mean, 4),
                                    'OLS_preds_RMSE' : round(OLS_preds_rmse, 4),
                                    'OLS_RMSE' : round(OLS_rmse, 4),
                                    'Tweedie_preds_RMSE':round(tweedie_preds_rmse, 4),
                                    'Tweedie_RMSE' : round(tweedie_rmse, 4)})
    
    return weather_val_df    





# df for weather RMSE results

def weather_rmse_test_df(y_df, rmse_test_mean, OLS_preds_rmse, OLS_rmse, tweedie_preds_rmse, tweedie_rmse):
    
    '''
    this function returns a df with the RMSE and baseline
    for the weather features
    '''

    # resetting y_val index 
    y_df.index.sort_values()
    y_df.reset_index(drop = True, inplace = True)

    # creating test results df
    weather_test_df = pd.DataFrame({'actual_y' : round(y_df, 4),
                                    'baseline' : round(rmse_test_mean, 4),
                                    'OLS_preds_RMSE' : round(OLS_preds_rmse, 4),
                                    'OLS_RMSE' : round(OLS_rmse, 4),
                                    'Tweedie_preds_RMSE':round(tweedie_preds_rmse, 4),
                                    'Tweedie_RMSE' : round(tweedie_rmse, 4)})
    
    return weather_test_df    





# df for weather RMSE results : val

def bar_rmse_val_df(y_df, rmse_val_mean, OLS_preds_rmse, OLS_rmse, tweedie_preds_rmse, tweedie_rmse):
    
    '''
    this function returns a df with the RMSE and baseline
    for the weather features
    '''

    # resetting y_val index 
    y_df.index.sort_values()
    y_df.reset_index(drop = True, inplace = True)

    # creating test results df
    bar_val_df = pd.DataFrame({'actual_y' : round(y_df, 4),
                                    'baseline' : round(rmse_val_mean, 4),
                                    'OLS_preds_RMSE' : round(OLS_preds_rmse, 4),
                                    'OLS_RMSE' : round(OLS_rmse, 4),
                                    'Tweedie_preds_RMSE':round(tweedie_preds_rmse, 4),
                                    'Tweedie_RMSE' : round(tweedie_rmse, 4)})
    
    return bar_val_df    





# df for bar RMSE results : test

def bar_rmse_test_df(y_df, rmse_test_mean, OLS_preds_rmse, OLS_rmse):
    
    '''
    this function returns a df with the RMSE and baseline
    for the weather features
    '''

    # resetting y_val index 
    y_df.index.sort_values()
    y_df.reset_index(drop = True, inplace = True)

    # creating test results df
    bar_test_df = pd.DataFrame({'actual_y' : round(y_df, 4),
                                    'baseline' : round(rmse_test_mean, 4),
                                    'OLS_preds_RMSE' : round(OLS_preds_rmse, 4),
                                    'OLS_RMSE' : round(OLS_rmse, 4)})
    
    return bar_test_df    




# plott weather results RMSEs

def plot_all_weather_RMSE(rmse_train_mean, OLS_preds_rmse, OLS_rmse, tweedie_preds_rmse, tweedie_rmse):
    
    toplot = ['OLS_preds_rmse', 'OLS_rmse', 'tweedie_preds_rmse', 'tweedie_rmse']

    plt.figure(figsize=(16,8))

    X_axis = np.arange(len(toplot))

    plt.axhline(rmse_train_mean, label = 'Baseline', color = 'magenta')
    # plt.bar( X_axis + 0.4, toplot, label = 'Tweedie RMSE', color = 'green')

    gure = [plt.bar(X_axis - 0.2, OLS_preds_rmse, 0.2, label = 'OLS RMSE predictions', color = 'lightgreen'),
            plt.bar(X_axis + 0.0, OLS_rmse, 0.2, label = 'OLS RMSE', color = 'lightpink'),
            plt.bar(X_axis + 0.2, tweedie_preds_rmse, 0.2, label = 'Tweedie RMSE predictions', color = 'lightblue'),
            plt.bar(X_axis + 0.4, tweedie_rmse, 0.2, label = 'Tweedie RMSE', color = 'green')]


    plt.ylim(0.47,  0.50)
    plt.xticks(X_axis, toplot)

    plt.xlabel('Model')
    plt.ylabel('RMSE')

    plt.title('RMSEs Of The OLS And Tweedie Models, Weather')
    plt.legend()
    plt.show()
    

    
def plot_all_bar_RMSE(rmse_train_mean, OLS_preds_rmse, OLS_rmse, tweedie_preds_rmse, tweedie_rmse):
    
    toplot = ['OLS_preds_rmse', 'OLS_rmse', 'tweedie_preds_rmse', 'tweedie_rmse']

    plt.figure(figsize=(16,8))

    X_axis = np.arange(len(toplot))

    plt.axhline(rmse_train_mean, label = 'Baseline', color = 'magenta')
    # plt.bar( X_axis + 0.4, toplot, label = 'Tweedie RMSE', color = 'green')

    gure = [plt.bar(X_axis - 0.2, OLS_preds_rmse, 0.2, label = 'OLS RMSE predictions', color = 'lightgreen'),
            plt.bar(X_axis + 0.0, OLS_rmse, 0.2, label = 'OLS RMSE', color = 'lightpink'),
            plt.bar(X_axis + 0.2, tweedie_preds_rmse, 0.2, label = 'Tweedie RMSE predictions', color = 'lightblue'),
            plt.bar(X_axis + 0.4, tweedie_rmse, 0.2, label = 'Tweedie RMSE', color = 'green')]


    plt.ylim(0.47,  0.50)
    plt.xticks(X_axis, toplot)

    plt.xlabel('Model')
    plt.ylabel('RMSE')

    plt.title('RMSEs Of The OLS And Tweedie Models, Bar-Visit Frequency')
    plt.legend()
    plt.show()
    
    

    
    
    
def plot_test_bar_RMSE(rmse_test_mean, OLS_preds_rmse, OLS_rmse):
    
    toplot = ['OLS_preds_rmse', 'OLS_rmse']

    plt.figure(figsize=(16,8))

    X_axis = np.arange(len(toplot))

    plt.axhline(rmse_test_mean, label = 'Baseline', color = 'magenta')
    # plt.bar( X_axis + 0.4, toplot, label = 'Tweedie RMSE', color = 'green')

    gure = [plt.bar(X_axis - 0.2, OLS_preds_rmse, 0.2, label = 'OLS RMSE predictions', color = 'lightgreen'),
            plt.bar(X_axis + 0.0, OLS_rmse, 0.2, label = 'OLS RMSE', color = 'lightpink')]


    plt.ylim(0.47,  0.50)
    plt.xticks(X_axis, toplot)

    plt.xlabel('Model')
    plt.ylabel('RMSE')

    plt.title('RMSEs Of The OLS And Tweedie Models, Bar-Visit Frequency')
    plt.legend()
    plt.show()
    
    
    

    
# decision tree with depth of 3    
    
def decision3(rmse_train_mean, X_train, y_train, target):
    
    '''
    this function intakes a df, X_train & y_train and 
    runs a decision tree on them with a max depth of 3.
    It outputs the accuracy.
    '''
    # setting the baseline for 'Y' to the train dataset mean
    baseline = rmse_train_mean
    
    print(f'The baseline of about {round(rmse_train_mean, 4)} indicates ' + 
          'the likelihood that a vehicle driver will accept the coupon.')

    # initialise the Decision Tree Classifier = clf
    seed = 23
    clf3 = DecisionTreeClassifier(max_depth = 3, random_state = seed)

    ### fitting the model : 
    clf3 = clf3.fit(X_train, y_train)

    # accurcy of the decision tree model
    print()
    print(f'Decision Tree Accuracy, max depth of 3 : {round(clf3.score(X_train, y_train), 4)}')
    
    return clf3
    
    
    
    
    
def random_decision3(rmse_train_mean, X_train, y_train, target):
    
    '''
    this function intakes a df, X_train & y_train and 
    runs a decision tree on them with a max depth of 3.
    It outputs the accuracy.
    '''
    # setting the baseline for 'Y' to the train dataset mean
    baseline = rmse_train_mean
    
    print(f'The baseline of about {round(rmse_train_mean, 4)} indicates ' + 
          'the likelihood that a vehicle driver will accept the coupon.')

    # initialise the Decision Tree Classifier = clf
    seed = 23
    clf3 = DecisionTreeClassifier(max_depth = 3, random_state = seed)

    ### fitting the model : 
    clf3 = clf3.fit(X_train, y_train)

    # accurcy of the decision tree model
    print()
    print(f'Decision Tree Accuracy, max depth of 3 : {round(clf3.score(X_train, y_train), 4)}')    

        
    # setting random forest classifier to 3 branches
    random = RandomForestClassifier(max_depth = 3, random_state = 23,
                           max_samples = 0.5)        
                            # 50pc of all observations will be placed into each random sample
        
    # training the random forest on the data
    random.fit(X_train, y_train)    

    # scoring the accuracy of the training set
    random.score(X_train, y_train)

    # accurcy of the decision tree model
    print()
    print(f'Random Forest Accuracy, max depth of 3 : {round(random.score(X_train, y_train), 4)}')
    
    return clf3, random
    
    
    
    
# def random_forest11(X_train, y_train):
    
#     '''
#     this function intakes X_train & y_train and 
#     runs a random forest on them with a max depth of 11.
#     It outputs the accuracy.
#     '''
    
#     # setting random forest classifier to 11 branches
#     random = RandomForestClassifier(max_depth = 11, random_state = 23,
#                            max_samples = 0.5)        
#                             # 50pc of all observations will be placed into each random sample
        
#     # training the random forest on the data
#     random.fit(X_train, y_train)    

#     # scoring the accuracy of the training set
#     random.score(X_train, y_train)

#     # accurcy of the decision tree model
#     print(f'Random Forest Accuracy, max depth of 11 : {round(random.score(X_train, y_train), 4)}')


    

# decision tree with depth of 7    
    
def decision_7(rmse_train_mean, X_train, y_train, target):
    
    '''
    this function intakes the rmse_train_mean, X_train & y_train and 
    runs a decision tree on them with a max depth of 7.
    It outputs the accuracy.
    '''
    # setting the baseline for 'Y' to the train dataset mean
    baseline = rmse_train_mean
    
    print(f'The baseline of about {round(rmse_train_mean, 4)} indicates ' + 
          'the likelihood that a vehicle driver will accept the coupon.')

    # initialise the Decision Tree Classifier = clf
    seed = 23
    clf7 = DecisionTreeClassifier(max_depth = 23, random_state = seed)

    ### fitting the model : 
    clf7 = clf7.fit(X_train, y_train)

    # accurcy of the decision tree model
    print()
    print(f'Decision Tree Accuracy, max depth of 7 : {round(clf7.score(X_train, y_train), 4)}')    
    
    return clf7
    

    
    
def random_decision_7(rmse_train_mean, X_train, y_train, target):
    
    '''
    this function intakes the rmse_train_mean, X_train & y_train and 
    runs a decision tree on them with a max depth of 17.
    It outputs the accuracy.
    '''
    # setting the baseline for 'Y' to the train dataset mean
    baseline = rmse_train_mean
    
    print(f'The baseline of about {round(rmse_train_mean, 4)} indicates ' + 
          'the likelihood that a vehicle driver will accept the coupon.')

    # initialise the Decision Tree Classifier = clf
    seed = 23
    clf7 = DecisionTreeClassifier(max_depth = 23, random_state = seed)

    ### fitting the model : 
    clf7 = clf7.fit(X_train, y_train)

    # accurcy of the decision tree model
    print()
    print(f'Decision Tree Accuracy, max depth of 7 : {round(clf7.score(X_train, y_train), 4)}')    

    
    # setting random forest classifier to 11 branches
    random7 = RandomForestClassifier(max_depth = 7, random_state = 23,
                           max_samples = 0.5)        
                            # 50pc of all observations will be placed into each random sample
        
    # training the random forest on the data
    random7.fit(X_train, y_train)    

    # scoring the accuracy of the training set
    random7.score(X_train, y_train)

    # accurcy of the decision tree model
    print()
    print(f'Random Forest Accuracy, max depth of 7 : {round(random7.score(X_train, y_train), 4)}')
    
    return clf7, random7

    
    
    
# def random_forest17(X_train, y_train):
    
#     '''
#     this function intakes X_train & y_train and 
#     runs a random forest on them with a max depth of 17.
#     It outputs the accuracy.
#     '''
    
#     # setting random forest classifier to 11 branches
#     random17 = RandomForestClassifier(max_depth = 17, random_state = 23,
#                            max_samples = 0.5)        
#                             # 50pc of all observations will be placed into each random sample
        
#     # training the random forest on the data
#     random17.fit(X_train, y_train)    

#     # scoring the accuracy of the training set
#     random17.score(X_train, y_train)

#     # accurcy of the decision tree model
#     print(f'Random Forest Accuracy, max depth of 17 : {round(random17.score(X_train, y_train), 4)}')





## classification report, train :

def classif_rpt_train(clf, X_df, y_df):
        y_predictions7 = clf.predict(X_df)
        print(f'Classification Report For Training Dataset \n  \n {classification_report(y_df, y_predictions7)}')

        
        
## classification report, val :

def classif_rpt_val(clf, X_df, y_df):
        y_predictions7 = clf.predict(X_df)
        print(f'Classification Report For Validate Dataset \n  \n {classification_report(y_df, y_predictions7)}')
        
        
        
## classification report, test :

def classif_rpt_test(clf, X_df, y_df):
        y_predictions7 = clf.predict(X_df)
        print(f'Classification Report For Test Dataset \n  \n {classification_report(y_df, y_predictions7)}')
        
        