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
from sklearn.linear_model import Ridge, Lasso, LassoLarsIC
from cleaning import *
from sklearn.feature_selection import VarianceThreshold

def run_model(model,X_train,X_test,y_train,y_test):
    print('Training R^2 :',model.score(X_train,y_train))
    y_pred_train = model.predict(X_train)
    print('Training Root Mean Square Error',np.sqrt(mean_squared_error(y_train,y_pred_train)))
    print('\n----------------\n')
    print('Testing R^2 :',model.score(X_test,y_test))
    y_pred_test = model.predict(X_test)
    print('Testing Root Mean Square Error',np.sqrt(mean_squared_error(y_test,y_pred_test)))

def compare_models(X_train, X_test, y_train, y_test):
    lr = LinearRegression()
    model_lr = lr.fit(X_train, y_train)
    run_model(model, X_train, X_test, y_train, y_test)
    ridge = Ridge()
    model_ridge = ridge.fit(X_train, y_train)
    run_model(ridge, X_train, X_test, y_train, y_test)
    lasso = Lasso()
    model_lasso = lasso.fit(X_train, X_test)
    run_model(model_lasso, X_train, X_test, y_train, y_test)

def compare_poly(degrees=2, X_train, X_test, y_train, y_test):
    for deg in range(degrees+1):
        poly = PolynomialFeatures(deg)
        X_poly_train = poly.fit_transform(X_train)
        X_poly_test = poly.fit(X_test)
        compare_models(X_poly_train, X_poly_test, y_train, y_test)


