from scratch_tree._base import Node, Stack
import numpy as np

class _BaseEstimator:
    def _node_creator(self, X, y, n_min=30, rf = False):
        first_stack = Stack(0, list(range(X.shape[0])), self._loss(y))
        stack_list, node_list = [first_stack] ,[]
        node_index = 0
        while len(stack_list)>0:
            stack = stack_list.pop()
            best_score, best_score_calc_result = self._node_opimizer(X, y, stack.data_set_list, rf, self._loss)
            right_set_mask = self._extract_mask_index(stack.data_set_list, best_score_calc_result.left_mask)
            left_set_mask = self._extract_mask_index(stack.data_set_list, best_score_calc_result.right_mask)

            if len(left_set_mask)<=1 or len(right_set_mask)<=1 or len(stack.data_set_list)<n_min:
                value = np.mean(y[stack.data_set_list])
                node_list.append(Node(stack.parent_index, -1, 0, stack.data_set_list,best_score,value))
            else:
                node_list.append(
                    Node(
                        stack.parent_index, 
                        best_score_calc_result.column_index, 
                        best_score_calc_result.divide_val, 
                        stack.data_set_list,
                        best_score,
                        0
                    ))
                stack_list.append(Stack(node_index, right_set_mask, best_score_calc_result.right_score))  
                stack_list.append(Stack(node_index, left_set_mask, best_score_calc_result.left_score))
            node_index += 1
        return self._add_node_att(node_list, node_index)
    
    def _boost_node_creator(self, X, y, n_min=30, max_div = 0):
        first_node = Node(0, -1, 0, list(range(X.shape[0])),self._loss(y),0)
        node_list = [first_node]
        score = first_node.score
        while len(node_list) <= 2 * max_div - 1:
            gain_list, calc_result_list = [], []
            for node_i in range(len(node_list)):
                if len(node_list[node_i].data_set) == 0 or node_list[node_i].var_index != -1:
                    gain_list.append(- np.inf)
                    calc_result_list.append(0)
                else:
                    best_score, best_score_calc_result = self._node_opimizer(
                        X, y, node_list[node_i].data_set, False, self._loss
                    )
                    gain_list.append(node_list[node_i].score - best_score)
                    calc_result_list.append(best_score_calc_result)

            best_i = gain_list.index(max(gain_list))
            calc_result = calc_result_list[best_i]

            node_list[best_i].threshold = best_score_calc_result.divide_val
            node_list[best_i].var_index = best_score_calc_result.column_index

            right_set_mask = self._extract_mask_index(node_list[best_i].data_set, calc_result.right_mask)
            right_value = np.mean(y[right_set_mask])
            right_node = Node(best_i, -1, 0, right_set_mask, node_list[best_i].score - gain_list[best_i], right_value)
            node_list.append(right_node)

            left_set_mask = self._extract_mask_index(node_list[best_i].data_set, calc_result.left_mask)
            left_value = np.mean(y[left_set_mask])
            left_node = Node(best_i, -1, 0, left_set_mask, node_list[best_i].score - gain_list[best_i], left_value)
            node_list.append(left_node)
        return self._add_node_att(node_list, len(node_list))
    