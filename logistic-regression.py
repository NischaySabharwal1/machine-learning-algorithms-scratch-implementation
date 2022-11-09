import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class Logisticregression:
  X,y,w, losses, prob = 0,0,0,[], 0

  def __init__(self, lr= 0.1, n_iter = 200, thresh = 0.5) -> None:
    self.lr = lr
    self.n_iter = n_iter
    self.thresh = thresh

  def sigmoid(self,z):
    return 1.0/(1.0+np.exp(-z))

  def predict(self,x,w):
    z = x.dot(w)
    return self.sigmoid(z)

  def loss(self,x,y,w):
    y_hat = self.predict(x,w)
    res1 = np.where(y_hat>0.00000000001, y_hat, -30)
    res2 = np.where((1-y_hat)>0.00000000001, 1-y_hat, -30)
    y_hat_log = np.log(res1, out = res1, where=res1>0)
    y_hat_log1 = np.log(res2, out = res2, where=res2>0)
    loss = (-1) * np.sum((y*y_hat_log) + ((1-y)*y_hat_log1))/len(y)
    return loss

  def derivative(self,x,y,w):
    y_hat = self.predict(x,w)
    return x.T.dot(y_hat-y)

  def check_array(self,X,y=pd.Series([0])):
    if type(X) == pd.DataFrame and type(y) == pd.Series:
      X = X.values
      y = y.values
    return X,y

  def append_ones(self,X):
    ones = np.ones(X.shape[0]).reshape(-1,1)
    return np.hstack((X, ones))

  def fit(self,x,y):
    x,y = self.check_array(x,y)
    self.X = x
    self.y = y
    x = self.preprocess(x)
    w = np.random.randn(x.shape[1])
    losslist, i = [], 0
    while True:
      i+=1
      losslist.append(self.loss(x,y,w))

      w = w - self.lr*self.derivative(x,y,w)
      
      if len(losslist)>2 and abs(losslist[-1]-losslist[-2]) < 1e-3:
        print(f'Algorithm converged in {i} steps')
        break
      if i==self.n_iter:
        print(f'Hard stop: {self.n_iter} iterations reached')
        break
    self.w = w
    self.losses = losslist
    self.prob = self.predict(self.preprocess(self.X), self.w)

  def standardize(self,X):
    scaler = StandardScaler().fit(self.X)
    return scaler.transform(X)
      
  def preprocess(self, X):
    X = self.standardize(X)
    X = self.append_ones(X)
    return X

  def probas(self,x):
    x, _ = self.check_array(x)
    if x.ndim == 1:
      x = x.reshape(1,-1)
    x = self.preprocess(x)
    y_hat = self.predict(x,self.w)

  def classify(self,x):
    x, _ = self.check_array(x)
    if x.ndim == 1:
      x = x.reshape(1,-1)
    x = self.preprocess(x)
    y_hat = self.predict(x,self.w)
    y_hat = [1 if i>self.thresh else 0 for i in y_hat]
    return y_hat

  def score(self,x,y):
    y_hat = self.classify(x)
    y = y.values
    c=0
    acc = np.mean([1 if y_hat[i]==y[i] else 0 for i in range(len(y_hat))])
    return acc

  def get_metrics(self,x,y):
    y_hat = self.classify(x)
    y = y.values
    tp,fp,tn,fn = 0,0,0,0
    for i in range(len(y)):
      if y_hat[i]==y[i]: 
        if y_hat[i] == 1: tp+=1
        else: tn+=1
      else:
        if y_hat[i]==1: fp+=1
        else: fn+=1
    tpr = tp/(tp+fn)
    tnr = tn/(tn+fp)
    fpr = fp/(fp+tn)
    fnr = fn/(tp+fn)
    precision = tp/(tp+fp)
    if precision<0.001:
      precision = 0.001
    elif tpr<0.001:
      tpr = 0.001
    f1 = 2/((1/precision)+(1/tpr))
    p = pd.DataFrame([precision, tpr, f1], index = ['Precision', 'Recall', 'F1-score'], columns = ['Scores'])
    return p
