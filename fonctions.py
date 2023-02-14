import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LassoLars
from sklearn.cluster import KMeans
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import TweedieRegressor
from math import sqrt
from scipy.stats import pearsonr, spearmanr
from scipy import stats

from sklearn.multioutput import MultiOutputClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier

from env import get_connection
import prepare


# turn off pink boxes for demo
import warnings
warnings.filterwarnings("ignore")

seed = 23



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
        
        
        
# plotting coupon acceptance in different weather

# def plot_weather(train):
    
#     '''
#     this function plots coupon acceptance based on different
#     types of weather. It takes in train dataset.
#     '''
    
#     g = sns.catplot(data = train, kind = 'bar', y = 'Y', x = 'weather3')

#     g.set_axis_labels('Snowy (0), Sunny (1), Rainy (2)', 'Liklihood Of Coupon Acceptance')

#     plt.grid()

#     plt.title('Liklihood Of Coupon Acceptance In Different Weather')
    
#     plt.show()
    
    

# plotting coupon acceptance for opposite direction   

def plot_opp_dir(train):
    
        
    '''
    this function plots coupon acceptance based on train data for
    people heading in the opposite direction
    '''
    
    g = sns.catplot(data = train, kind = 'bar', y = 'Y', x = 'dir_opp', hue = 'weather')

    g.set_axis_labels('', 'Liklihood Of Coupon Acceptance')

    plt.grid()

    plt.title('Likelihood Of Coupon Acceptance When Heading In The Opposite Direction In Different Weather')
    
    plt.show()
    

    
# plotting coupon acceptance based on bar visits

def plot_bar(train):
    
    '''
    this function plots bar-visit frequency based on train data
    '''
    
    g = sns.catplot(data = train, kind = 'bar', y = 'Y', x = 'bar')

    g.set_axis_labels('', 'Liklihood Of Coupon Acceptance')

    plt.grid()

    plt.title('Liklihood Of Coupon Based On Frequency Of Bar Visits')
    
    plt.show()
    
    
    
    
# plotting income groups and coupon acceptance
    
def plot_inc(train):   
       
    '''
    this function plots income groups based on train data
    and visualises how they compare to coupon acceptance.S
    '''
    
    plt.figure().set_figwidth(15)

    g = sns.catplot(data = train, kind = 'bar', y = 'Y', x = 'income')

    g.set_axis_labels('', 'Liklihood Of Coupon Acceptance')

    plt.grid()

    plt.xticks(rotation = 45)

    plt.title('Likelihood Of Coupon Acceptance Based On Income Group')

    plt.show()    
    
    
    