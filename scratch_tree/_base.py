from abc import ABCMeta, abstractmethod
import numpy as np
from collections import namedtuple

class BaseTree(metaclass=ABCMeta):
    @abstractmethod
    def fit(self):
        pass
    @abstractmethod
    def predict(self):
        pass
        

class Stack:
    def __init__(self, parent_index, data_set_list, score):
        self.parent_index = parent_index
        self.data_set_list = data_set_list
        self.score = score

class Node:
    def __init__(self, parent_index, var_index, threshold, data_set, score, value):
        self.parent_index = parent_index
        self.var_index = var_index
        self.threshold = threshold
        self.data_set=data_set
        self.score = score
        self.value = value
        self.left_child = 0
        self.right_child = 0
        

class _Tree():
        
    def _choose_best_score(self, best_score_list, calc_result_list):
        best_score = min(best_score_list)
        best_score_index = best_score_list.index(best_score)
        best_score_calc_result = calc_result_list[best_score_index]
        return best_score, best_score_calc_result
    
    
    def _add_tuple(self, column_index,divide_val, left_score,
                   right_score,left_mask,right_mask
                  ):
        calc_result = namedtuple('calc_result' , 
                            [
                                'column_index',
                                'divide_val',
                                'left_score',
                                'right_score',
                                'left_mask',
                                'right_mask'
                            ])
        return calc_result(column_index, divide_val, left_score, right_score, left_mask, right_mask)
    
    def _make_data_from_mask(self, original_data_set, mask):
        masked_data = np.ma.MaskedArray(original_data_set, mask=~mask, fill_value=None)
        masked_data = np.ma.filled(masked_data.astype(float), np.nan)
        return masked_data
    
    def _make_branch_dataset(self, col, y):
        boadcast, boadcast_trainspose = np.meshgrid(col, col)
        left_mask = (boadcast < boadcast_trainspose)
        right_mask = ~left_mask
        y_tiled = np.tile(y,(len(y),1))

        left_set = self._make_data_from_mask(y_tiled, left_mask)
        right_set = self._make_data_from_mask(y_tiled, right_mask)
        return left_set, right_set, left_mask, right_mask
    
    def _node_opimizer(self, X, y, mask, rf, loss):
        calc_result_list,best_score_list = [],[]
        if rf:
            pick_col = int(X.shape[1]/2)
            num_columns = np.random.choice(X.shape[1],pick_col,replace=False)
        else:
            num_columns = range(X.shape[1])

        for column_index in num_columns:
            column_values,target_value = X[mask,column_index], y[mask]
            left_data, right_data, left_mask, right_mask = self._make_branch_dataset(column_values,target_value)
            if len(left_data) == 0 or len(right_data)==0: 
                continue
            left_score_arr, right_score_arr = loss(left_data), loss(right_data)
            tot_score_arr = left_score_arr + right_score_arr
            best_score = np.amin(tot_score_arr, where=~np.isnan(tot_score_arr),initial= np.inf)
            if best_score == np.nan or best_score == np.inf:
                continue
            best_score_index = np.where(tot_score_arr==best_score)[0][0]
            calc_result = self._add_tuple(
                column_index,
                column_values[best_score_index],
                left_score_arr[best_score_index],
                right_score_arr[best_score_index],
                left_mask[best_score_index], 
                right_mask[best_score_index]
            )
            calc_result_list.append(calc_result)
            best_score_list.append(best_score)
        return self._choose_best_score(best_score_list, calc_result_list)
    
    def _extract_mask_index(self, data_set_list, sub_set_mask):
        return np.array(data_set_list)[[sub_set_mask]]
    
    def _add_node_att(self, node, node_index):
        for h in range(node_index - 1, 0, -1):
            pa = node[h].parent_index
            if node[pa].left_child == 0:
                node[pa].left_child = h
            else:
                node[pa].right_child = h
        return node
    
    def _make_prediction(self, X_arr, node):
        predict_arr = []
        for X in X_arr:
            predict_arr.append(self._value(X, node))
        return np.array(predict_arr)
    
    def _value(self, data, node):
        r = 0
        while node[r].var_index != -1:
            if data[node[r].var_index] < node[r].threshold:
                r = node[r].left_child
            else:
                r = node[r].right_child
        return node[r].value