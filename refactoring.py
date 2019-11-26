import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols



def run_model(model, X_train, X_test, y_train, y_test):
    print('Training R^2 :',round(model.score(X_train, y_train),4))
    y_pred_train = model.predict(X_train)
    print('Training Root Mean Square Error',round(np.sqrt(mean_squared_error(y_train, y_pred_train)),4))
    print('\n----------------\n')
    print('Testing R^2 :',round(model.score(X_test, y_test),4))
    y_pred_test = model.predict(X_test)
    print('Testing Root Mean Square Error',round(np.sqrt(mean_squared_error(y_test, y_pred_test)),4))
    qq_plot(model, pd.concat([X_train, X_test]), pd.concat([y_train, y_test]))
    
def compare_poly(X_train, X_test, y_train, y_test, degrees=2):
    for deg in range(2,degrees+1):
        poly = PolynomialFeatures(deg)
        X_poly_train = pd.DataFrame(poly.fit_transform(X_train))
        X_poly_test = pd.DataFrame(poly.fit_transform(X_test))
        lr = LinearRegression()
        model_lr = lr.fit(X_poly_train, y_train)
        run_model(model_lr, X_poly_train, X_poly_test, y_train, y_test)

def residuals(model, X, y):
    y_preds = model.predict(X)
    return y - y_preds

def qq_plot(model, X, y):
    res = residuals(model, X, y)
    fig = sm.graphics.qqplot(res, dist = stats.norm, line= '45', fit= True)
    
def ols_model(X, y, target):
    outcome = target
    predictors = '+'.join(X.columns)
    formula = outcome + '~' + predictors
    model = ols(formula=formula, data=pd.concat([X,y], axis = 1)).fit()
    return model

def cols_to_drop(X):
    high_corrs = (np.sum(np.abs(X.corr()) > 0.4) > 2)
    cols_to_drop = list(high_corrs[high_corrs == True].index)
    return cols_to_drop
    
def drop_high_corrs(X):
    while True:
        cols = cols_to_drop(X)
        if not cols:
            break
        else:
            X.drop(cols.pop(0), axis= 1, inplace=True)
    return X

def var_check(model, X ,y):
    plt.figure(figsize = (15,8))
    plt.scatter(model.predict(X), model.resid)
    plt.plot(model.predict(X), [0 for i in range(len(X))])
    plt.ylabel('Residual')
    plt.title('Homoscedasticity check')
    plt.show()