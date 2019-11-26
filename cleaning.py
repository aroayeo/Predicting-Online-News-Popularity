import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from itertools import combinations
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import *
from sklearn.linear_model import Lasso
from cleaning import *
from sklearn.feature_selection import VarianceThreshold
from sklearn import metrics
from sklearn.preprocessing import Imputer

    
def strip_columns(df):
    df = df.rename(columns=lambda x: x.strip())
    return df

def log_target(data, target):
    data['log_' + target]=np.log(data[target])
    sns.distplot(data['log_' + target])
    plt.title('Target Variable Distribution')
    plt.show()
    return data

def variables(df,a_list):
    df = df[a_list]
    return df

def delete_value(df, col):
    delete = df.loc[(df[col] == 701)]
    df.drop(delete.index, axis=0, inplace=True)
    return df

def scale(x,y):
    x_scale = x.apply(lambda x: (x - np.min(x))/(np.max(x)-np.min(x)))
    df_scale = pd.concat([x_scale, y], axis=1)
    return df_scale

def channel_graph(data, channel_list):
    
    channel = data[channel_list]

    channel = channel.rename(columns={'data_channel_is_lifestyle':'lifestyle', 'data_channel_is_entertainment':'entertainment', 
                                  'data_channel_is_bus':'business','data_channel_is_socmed':'social_media', 
                                  'data_channel_is_tech':'tech','data_channel_is_world':'world'})
    channel_topic = pd.Series(channel.columns[np.where(channel!=0)[1]])
    channel = pd.concat([data['shares'], channel_topic], axis=1)
    channel = channel.rename(columns={0:'channel'})
    group = channel.groupby('channel').sum()
    sns.barplot(x=group.index, y=group['shares'], palette='Greens_d').set_title('Number of shares by channel')
    plt.xticks(rotation=90)
    plt.show()
    

def weekday_graph(data, weekdays_list):
    data = pd.read_csv('OnlineNewsPopularity.csv')
    data = strip_columns(data)
    weekday = data[weekdays_list]
    weekday = weekday.rename(columns={'weekday_is_monday':'Monday', 'weekday_is_tuesday':'Tuesday',
       'weekday_is_wednesday':'Wednesday', 'weekday_is_thursday':'Thursday', 'weekday_is_friday':'Friday',
       'weekday_is_saturday':'Saturday', 'weekday_is_sunday':'Sunday'})
    weekday_shares = pd.Series(weekday.columns[np.where(weekday!=0)[1]])
    weekday_shares = pd.concat([data['shares'], weekday_shares], axis=1)
    weekday_shares = weekday_shares.rename(columns={0:'week_day'})
    week_g = weekday_shares.groupby('week_day').sum()
    week = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    sns.barplot(x=week_g.index, y=week_g['shares'], palette="Blues_d", order=week).set_title('Number of shares by day of the week')
    plt.xticks(rotation=90)
    plt.show()
    
