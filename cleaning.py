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
    data[target]=np.log(data[target])
    sns.distplot(data[target])
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
