from scratch_tree._base import BaseTree, _Tree
from scratch_tree._base_estimator import _BaseEstimator

import numpy as np

class ScratchDecisionTreeRegressor(BaseTree, _Tree, _BaseEstimator):
    def __init__(self, n_min = 30):
        self.n_min = n_min
    
    def _calc_loss(self, x):
        x_mean=np.nanmean(x)
        x = x[~np.isnan(x)]
        return(np.linalg.norm(x-x_mean)**2)
    
    def _loss(self, y):
        if len(y.shape)==1:
            y = y.reshape(1,-1)
        return np.apply_along_axis(
            lambda x: self._calc_loss(x), 1, y
        )
        
    
    def fit(self, X_train, y_train, n_min = 30):
        self.node = self._node_creator(X_train,y_train, n_min = self.n_min)
    
    def predict(self, X_test):
        return self._make_prediction(X_test, self.node)
    
    
    
    
class ScratchRandomforestRegressor(BaseTree, _Tree, _BaseEstimator):
    def __init__(self, n_min = 30, n_estimator= 100):
        self.n_min = n_min
        self.n_estimator = n_estimator
    
    def _calc_loss(self, x):
        x_mean=np.nanmean(x)
        x = x[~np.isnan(x)]
        return(np.linalg.norm(x-x_mean)**2)
    
    def _loss(self, y):
        if len(y.shape)==1:
            y = y.reshape(1,-1)
        return np.apply_along_axis(
            lambda x: self._calc_loss(x), 1, y
        )
    
    def fit(self, X_train, y_train):
        self.trees_list = []
        for _ in range(self.n_estimator):
            data_mask = np.random.choice(len(X_train), int(len(X_train)*5/5), replace=True)
            X_train_t = X_train[data_mask]
            y_train_t = y_train[data_mask]
            self.trees_list.append(self._node_creator(X_train_t, y_train_t, n_min = self.n_min, rf = True))
    
    def predict(self, X_test):
        pre_arr = np.array([])
        for i in range(len(self.trees_list)):
            pre = self._make_prediction(X_test, self.trees_list[i])
            if i == 0:
                pre_arr = pre
            else:
                pre_arr = np.vstack((pre_arr, pre))
        
        pre = pre_arr.sum(axis=0) / pre_arr.shape[0]
        return pre
    
    
    
    
class ScratchGradientBoostingRegressor(BaseTree, _Tree, _BaseEstimator):
    def __init__(self, n_min = 30, n_estimator=150, lamda= 0.1, max_div = 3):
        self.n_min = n_min
        self.n_estimator = n_estimator
        self.lamda = 0.1
        self.max_div = 3
    
    def _calc_loss(self, x):
        x_mean=np.nanmean(x)
        x = x[~np.isnan(x)]
        return(np.linalg.norm(x-x_mean)**2)
    
    def _loss(self, y):
        if len(y.shape)==1:
            y = y.reshape(1,-1)
        return np.apply_along_axis(
            lambda x: self._calc_loss(x), 1, y
        )
    
    def fit(self, X_train, y_train):
        self.trees_list = []
        remain_y = y_train

        for b in range(self.n_estimator):
            self.trees_list.append(self._boost_node_creator(X_train, remain_y, max_div = self.max_div))
            pre = self._make_prediction(X_train, self.trees_list[b])
            remain_y = remain_y - self.lamda * pre
    
    
    def predict(self, X_test):
        for i in range(self.n_estimator):
            pre = self.lamda * self._make_prediction(X_test, self.trees_list[i])
            if i == 0:
                pre_arr = pre
            else:
                pre_arr = np.vstack((pre_arr, pre))
        pre = pre_arr.sum(axis=0)
        return pre