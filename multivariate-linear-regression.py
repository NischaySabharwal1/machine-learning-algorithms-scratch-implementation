import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from tabulate import tabulate
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

class Linearregression():
  X,y,w, loss = 0,0,0,[]
  def predict(self,X,w):
    return np.dot(X,w)
  def error(self,X,w, y):
    y_hat = self.predict(X,w)
    e = y-y_hat
    return e
  def mse(self,X,w,y):
    err = self.error(X,w,y)**2
    n = len(y)
    mser = np.sum(err)/n
    return mser
  def derivative(self,X,w,y):
    e = (-1)*self.error(X,w,y)
    xt = X.T
    n = len(y)
    derw = np.dot(xt, e)
    derw = derw * (2/n)
    return derw
  def step(self,X,w,y, lr):
    derw = self.derivative(X,w,y)
    w = w - lr*derw
    return w
  def check_array(self,X,y):
    if type(X) == pd.DataFrame and type(y) == pd.Series:
      X = X.values
      y = y.values
    return X,y
  def fit(self,X,y,n_iter:int=200,lr:float=0.1):
    X,y = self.check_array(X,y)
    self.X = X
    self.y = y
    mser = []
    w = np.random.randn(X.shape[1])
    mser.append(self.mse(X,w,y))
    i=0
    while True:
      i+=1
      w = self.step(X,w,y, lr)
      mser.append(self.mse(X,w,y))
      if len(mser)>2 and abs(mser[-1]-mser[-2]) < 1e-7:
        print(f'Algorithm converged in {i} iterations')
        break
      elif i == n_iter:
        print(f'Hard Stop: {n_iter} iterations exceeded')
        break
    self.loss = mser
    self.w = w
  def r2_score(self,X, y):
    X,y = self.check_array(X,y)
    y_hat = self.predict(X, self.w)
    y_mean = np.mean(y)
    e2 = y-y_hat
    e1 = y - y_mean
    r2 = 1-(np.sum(e2**2)/np.sum(e1**2))
    return r2