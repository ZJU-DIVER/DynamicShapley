#!/usr/bin/env python
# coding: utf-8

# # Dynamic Shapley Value Computation Examples
# 
# ## Environment
# 
# Prepare the environment for experiments.

# In[ ]:


import random

import numpy as np
from sklearn import svm

from .data_utils import (
    load_tabular_data, preprocess_data, variance, normalize, save_npy, load_npy, comp
)
import dynashap


# ## Part 0 Prepare Data
# 
# In this part, we will prepare the data for demonstrating the algorithms. We give a toy size testing case whose time cost is pretty small.

# | item | description |
# | - | - |
# | data points | 20 -> 21/22/19/18 |
# | data set | iris (4F3C) |
# | model | SVM |

# In[ ]:


# Prepare the train index and valid index for iris dataset
dict_no = dict()

train_num = 20

train_index = []
test_index = []

last_num = train_num
for j in range(3):
    c_num = int((train_num + 1) / 3)
    train_index += random.sample([i for i in range(50 * j, 50 * (j + 1))], min([last_num, c_num]))
    last_num = last_num - c_num
    
test_index = list(set(i for i in range(3 * 50)) - set(train_index))

dict_no['train'] = train_index
dict_no['test'] = test_index

load_tabular_data('iris', dict_no, 'train_20p_3c.csv', 'test_20p_3c.csv')


# In[ ]:


x_train, y_train, x_test, y_test, columns_name = preprocess_data('train_20p_3c.csv', 'test_20p_3c.csv')
model = svm.SVC(decision_function_shape='ovo')

plus_time = 500
normal_time = 50


# In[ ]:


reserved_x, reserved_y = x_test[-2:], y_test[-2:]
x_test, y_test = x_test[:-2], y_test[:-2]


# In[ ]:


mc_plus_sv_20 = dynashap.mc_shap(x_train, y_train, x_test, y_test, model,
                                 len(y_train) * plus_time)
save_npy('mc_plus_sv_20.npy', mc_plus_sv_20)


# ## Part 1 Add single point
# 
# ### 1.1 Given Shapley value

# In[ ]:


added_x_train = np.append(x_train, [reserved_x[0]], axis=0)
added_y_train = np.append(y_train, reserved_y[0])


# In[ ]:


mc_plus_sv_21 = dynashap.mc_shap(added_x_train, added_y_train, x_test, y_test, model,
                                 len(added_y_train) * plus_time)
save_npy('mc_plus_sv_21.npy', mc_plus_sv_21)


# ### 1.2 Computation
# 
# ---
# Algorithm list:
# 
# * Baseline SV
# * Delta SV
# * Pivot SV
# * Heuristic SV
# * Monte Carlo SV
# ---

# In[ ]:


init_sv = load_npy('mc_plus_sv_20.npy')


# In[ ]:


# Baseline
base_shap = dynashap.BaseShap(x_train, y_train, x_test, y_test, model, init_sv)

params = {
    'method': 'avg'
}
base_avg_sv_21 = base_shap.add_single_point(reserved_x[0], reserved_y[0], params=params)
save_npy('base_avg_sv_21.npy', base_avg_sv_21)

params = {
    'method': 'loo'
}
base_loo_sv_21 = base_shap.add_single_point(reserved_x[0], reserved_y[0], params=params)
save_npy('base_loo_sv_21.npy', base_loo_sv_21)

# Delta
delta_shap = dynashap.DeltaShap(x_train, y_train, x_test, y_test, model, init_sv)
delta_sv_21 = delta_shap.add_single_point(reserved_x[0], reserved_y[0],
                                          len(added_y_train) * normal_time)
save_npy('delta_sv_21.npy', delta_sv_21)


# In[ ]:


# Pivot
# pivot_shap = dynashap.PivotShap(x_train, y_train, x_test, y_test, model, None)
# pivot_shap.prepare(len(y_train) * normal_time, proc_num=1)
# pivot_sv_d_21 = pivot_shap.add_single_point(reserved_x[0], reserved_y[0],
#                             m=len(added_y_train) * normal_time, proc_num=1, params={'method': 'd'})
# save_npy('pivot_sv_d_21.npy', pivot_sv_d_21)

pivot_shap = dynashap.PivotShap(x_train, y_train, x_test, y_test, model, None)
pivot_shap.prepare(len(added_y_train) * normal_time, 2)
pivot_sv_s_21 = pivot_shap.add_single_point(reserved_x[0], reserved_y[0],
                            proc_num=2, params={'method': 's'})
save_npy('pivot_sv_s_21.npy', pivot_sv_s_21)


# In[ ]:


print(pivot_shap.permutations[0])
print(pivot_shap.permutations[525])


# In[ ]:


# Heuristic
params = {
    'method': 'knn'
}
heur_shap = dynashap.HeurShap(x_train, y_train, x_test, y_test, model, init_sv, params=params)
params = {
    'n_neighbors': 5
}
heur_shap.prepare(params=params)
heur_knn_sv_21 = heur_shap.add_single_point(reserved_x[0], reserved_y[0])
save_npy('heur_knn_sv_21.npy', heur_knn_sv_21)

params = {
    'method': 'knn+'
}
heur_shap = dynashap.HeurShap(x_train, y_train, x_test, y_test, model, init_sv, params=params)
flags = {
    'exact': False, 
    'train': True
}
params = {
    'n_neighbors': 3,
    'simi_type': 'ed',
    'f_shap': 'n*n', 
    'rela': ['poly', 1],
    'train_idxs': [3, 11, 18],
    'm': (len(y_train) - 1) * plus_time
}
heur_shap.prepare(flags=flags, params=params)
heur_knn_plus_sv_21 = heur_shap.add_single_point(reserved_x[0], reserved_y[0])
save_npy('heur_knn_plus_sv_21.npy', heur_knn_plus_sv_21)

# Monte Carlo
mc_sv_21 = dynashap.mc_shap(added_x_train, added_y_train, x_test, y_test, model,
                            len(added_y_train) * normal_time)
save_npy('mc_sv_21.npy', mc_sv_21)


# ### 1.3 Comparison

# In[ ]:


# Load data
mc_plus_sv_21  = load_npy('mc_plus_sv_21.npy')

base_avg_sv_21 = load_npy('base_avg_sv_21.npy')
base_loo_sv_21 = load_npy('base_loo_sv_21.npy')
delta_sv_21    = load_npy('delta_sv_21.npy')
pivot_sv_d_21  = load_npy('pivot_sv_d_21.npy')
pivot_sv_s_21  = load_npy('pivot_sv_s_21.npy')
pivot_sv_21     = load_npy('pivot_sv_21.npy')
knn_sv_21      = load_npy('heur_knn_sv_21.npy')
knn_plus_sv_21 = load_npy('heur_knn_plus_sv_21.npy')
mc_sv_21       = load_npy('mc_sv_21.npy')


# In[ ]:


comp(mc_plus_sv_21, base_avg_sv_21, 'base avg')
comp(mc_plus_sv_21, base_loo_sv_21, 'base loo')
comp(mc_plus_sv_21, knn_sv_21, 'knn')
comp(mc_plus_sv_21, knn_plus_sv_21, 'knn+')
comp(mc_plus_sv_21, delta_sv_21, 'delta')
comp(mc_plus_sv_21, pivot_sv_d_21, 'pivot diff')
comp(mc_plus_sv_21, pivot_sv_s_21, 'pivot same')
comp(mc_plus_sv_21, mc_sv_21, 'mc')


# ## Part 2 Add multiple points

# In[ ]:


added_x_train = np.append(x_train, reserved_x, axis=0)
added_y_train = np.append(y_train, reserved_y)


# ### 2.1 Given Shapley value

# In[ ]:


mc_plus_sv_22 = dynashap.mc_shap(added_x_train, added_y_train, x_test, y_test, model,
                                 len(added_y_train) * plus_time)
save_npy('mc_plus_sv_22.npy', mc_plus_sv_22)


# In[ ]:


mc_sv_22 = dynashap.mc_shap(added_x_train, added_y_train, x_test, y_test, model,
                            len(added_y_train) * normal_time)
save_npy('mc_sv_22.npy', mc_sv_22)


# ### 2.2 Computation
# 
# ---
# Algorithm list:
# 
# * Baseline SV
# * Delta SV
# * Heuristic SV
# * Monte Carlo SV
# ---

# In[ ]:


init_sv = load_npy('mc_plus_sv_20.npy')

# Baseline
base_shap = dynashap.BaseShap(x_train, y_train, x_test, y_test, model, init_sv)

params = {
    'method': 'avg'
}
base_avg_sv_22 = base_shap.add_multi_points(reserved_x, reserved_y, params=params)
save_npy('base_avg_sv_22.npy', base_avg_sv_22)

params = {
    'method': 'loo'
}
base_loo_sv_22 = base_shap.add_multi_points(reserved_x, reserved_y, params=params)
save_npy('base_loo_sv_22.npy', base_loo_sv_22)

# Delta
delta_sv_21 = load_npy('delta_sv_21.npy')
delta_shap = dynashap.DeltaShap(np.append(x_train, [reserved_x[0]], axis=0), 
                                np.append(y_train, reserved_y[0]), 
                                x_test, y_test, model, delta_sv_21)
delta_sv_22 = delta_shap.add_single_point(reserved_x[1], reserved_y[1], len(added_y_train) * normal_time)
save_npy('delta_sv_22.npy', delta_sv_22)

# Pivot
pivot_shap = dynashap.PivotShap(x_train, y_train, x_test, y_test, model, init_sv)
pivot_shap.prepare(len(y_train) * plus_time)
pivot_shap.add_single_point(reserved_x[0], reserved_y[0], (len(y_train) + 1) * normal_time, flags={'flag_lsv': True})
pivot_sv_22 = pivot_shap.add_single_point(reserved_x[1], reserved_y[1], len(added_y_train) * normal_time, flags={'flag_lsv': True})
save_npy('pivot_sv_22.npy', pivot_sv_22)

# Heuristic
params = {
    'method': 'knn'
}
heur_shap = dynashap.HeurShap(x_train, y_train, x_test, y_test, model, init_sv, params=params)
params = {
    'n_neighbors': 4
}
heur_shap.prepare(params=params)
heur_knn_sv_22 = heur_shap.add_multi_points(reserved_x, reserved_y)
save_npy('heur_knn_sv_22.npy', heur_knn_sv_22)

params = {
    'method': 'knn+'
}
heur_shap = dynashap.HeurShap(x_train, y_train, x_test, y_test, model, init_sv, params=params)
flags = {
    'exact': False, 
    'train': True
}
params = {
    'n_neighbors': 4,
    'simi_type': 'ed',
    'f_shap': 'n*n', 
    'rela': ['poly', 1],
    'train_idxs': [3, 11, 18],
    'm': (len(y_train) - 1) * plus_time
}
heur_shap.prepare(flags=flags, params=params)
heur_knn_plus_sv_22 = heur_shap.add_multi_points(reserved_x, reserved_y)
save_npy('heur_knn_plus_sv_22.npy', heur_knn_plus_sv_22)

# Monte Carlo
mc_sv_22 = dynashap.mc_shap(added_x_train, added_y_train, x_test, y_test, model, len(added_y_train) * normal_time)
save_npy('mc_sv_22.npy', mc_sv_22)


# ### 2.3 Comparison

# In[ ]:


# Load
mc_plus_sv_22  = load_npy('mc_plus_sv_22.npy')

base_avg_sv_22 = load_npy('base_avg_sv_22.npy')
base_loo_sv_22 = load_npy('base_loo_sv_22.npy')
delta_sv_22    = load_npy('delta_sv_22.npy')
pivot_sv_22     = load_npy('pivot_sv_22.npy')
knn_sv_22      = load_npy('heur_knn_sv_22.npy')
knn_plus_sv_22 = load_npy('heur_knn_plus_sv_22.npy')
mc_sv_22       = load_npy('mc_sv_22.npy')


# In[ ]:


# Variance
comp(mc_plus_sv_22, base_avg_sv_22, 'base avg')
comp(mc_plus_sv_22, base_loo_sv_22, 'base loo')
comp(mc_plus_sv_22, knn_sv_22, 'knn')
comp(mc_plus_sv_22, knn_plus_sv_22, 'knn+')
comp(mc_plus_sv_22, delta_sv_22, 'delta')
comp(mc_plus_sv_22, pivot_sv_22, 'pivot')
comp(mc_plus_sv_22, mc_sv_22, 'mc')


# ## Part 3 Delete single point

# In[ ]:


# delete one point
del_point_idx = 19
del_idxs = [19]
deleted_idxs = np.delete(np.arange(len(y_train)), del_idxs)

deleted_x_train = x_train[deleted_idxs]
deleted_y_train = y_train[deleted_idxs]


# ### 3.1 Given Shapley value

# In[ ]:


mc_plus_sv_19 = dynashap.mc_shap(deleted_x_train, deleted_y_train, x_test, y_test, model,
                                 len(deleted_y_train) * plus_time)
save_npy('mc_plus_sv_19.npy', mc_plus_sv_19)


# ### 3.2 Computation
# 
# ---
# Algorithm list:
# 
# * Delta SV
# * Heuristic SV
# * Monte Carlo SV
# ---

# In[ ]:


# Delta

delta_shap = dynashap.DeltaShap(x_train, y_train, x_test, y_test,
                                model, init_sv)
delta_sv_19 = delta_shap.del_single_point(del_point_idx,
                                          len(deleted_x_train) * normal_time)
save_npy('delta_sv_19.npy', delta_sv_19)

# Heuristic
params = {
    'method': 'knn'
}
heur_shap = dynashap.HeurShap(x_train, y_train, x_test, y_test, model, init_sv, params=params)
params = {
    'n_neighbors': 4
}
heur_shap.prepare(params=params)
heur_knn_sv_19 = heur_shap.del_single_point(del_point_idx)
save_npy('heur_knn_sv_19.npy', heur_knn_sv_19)

params = {
    'method': 'knn+'
}
heur_shap = dynashap.HeurShap(x_train, y_train, x_test, y_test, model, init_sv, params=params)
flags = {
    'exact': False,
    'train': True
}
params = {
    'n_neighbors': 4,
    'simi_type': 'ed',
    'f_shap': 'n*n',
    'rela': ['poly', 1],
    'train_idxs': [3, 11, 16],
    'm': (len(y_train) - 1) * plus_time
}
heur_shap.prepare(flags=flags, params=params)
heur_knn_plus_sv_19 = heur_shap.del_single_point(del_point_idx)
save_npy('heur_knn_plus_sv_19.npy', heur_knn_plus_sv_19)

# MC

mc_sv_19 = dynashap.mc_shap(deleted_x_train, deleted_y_train, x_test, y_test, model,
                            len(deleted_y_train) * normal_time)
save_npy('mc_sv_19.npy', mc_sv_19)


# ### 3.3 YN Shap Check
# 
# > This case is just used to proof that YN-NN is `ZERO ERROR`, which means that YN-NN algorithm will
# > not bring new error into the Shapley value.

# In[ ]:


yn_shap = dynashap.YnShap(x_train, y_train, x_test, y_test,
                          model, init_sv)
flags = {'exact': True}
yn_shap.prepare(1, flags)
yn_sv_19 = yn_shap.del_single_point(del_point_idx, flags)
save_npy('yn_sv_19.npy', yn_sv_19)

exact_sv_19 = dynashap.exact_shap(deleted_x_train, deleted_y_train, x_test, y_test, model)
save_npy('exact_sv_19.npy', exact_sv_19)


# In[ ]:


print('The variance between yn_sv and exact_sv: \t %f' %
      variance(exact_sv_19, normalize(exact_sv_19, yn_sv_19)))


# ### 3.4 Comparison

# In[ ]:


# Load
mc_plus_sv_19  = load_npy('mc_plus_sv_19.npy')

delta_sv_19    = load_npy('delta_sv_19.npy')
knn_sv_19      = load_npy('heur_knn_sv_19.npy')
knn_plus_sv_19 = load_npy('heur_knn_plus_sv_19.npy')
mc_sv_19       = load_npy('mc_sv_19.npy')


# In[ ]:


# Variance
comp(mc_plus_sv_19, delta_sv_19, 'delta')
comp(mc_plus_sv_19, mc_sv_19, 'mc')
comp(mc_plus_sv_19, knn_sv_19, 'knn')
comp(mc_plus_sv_19, knn_plus_sv_19, 'knn+')


# ## Part 4 Delete multiple points

# In[ ]:


# delete two points
del_idxs = [18, 19]
deleted_idxs = np.delete(np.arange(len(y_train)), del_idxs)

deleted_x_train = x_train[deleted_idxs]
deleted_y_train = y_train[deleted_idxs]


# ### 4.1 Given Shapley value

# In[ ]:


mc_plus_sv_18 = dynashap.mc_shap(deleted_x_train, deleted_y_train, x_test, y_test, model,
                                 len(deleted_y_train) * plus_time)
save_npy('mc_plus_sv_18.npy', mc_plus_sv_18)


# ### 4.2 Computation

# ---
# Algorithm list:
# 
# * Delta SV
# * Heuristic SV
# * Monte Carlo SV
# ---

# In[ ]:


# Delta
delta_shap = dynashap.DeltaShap(x_train[:19], y_train[:19], x_test, y_test,
                                model, delta_sv_19)
delta_sv_18 = delta_shap.del_single_point(del_idxs[0],
                                          len(deleted_x_train) * normal_time)
save_npy('delta_sv_18.npy', delta_sv_18)

# Heuristic
params = {
    'method': 'knn'
}
heur_shap = dynashap.HeurShap(x_train, y_train, x_test, y_test, model, init_sv, params=params)
params = {
    'n_neighbors': 4
}
heur_shap.prepare(params=params)
heur_knn_sv_18 = heur_shap.del_multi_points(del_idxs)
save_npy('heur_knn_sv_18.npy', heur_knn_sv_18)

params = {
    'method': 'knn+'
}
heur_shap = dynashap.HeurShap(x_train, y_train, x_test, y_test, model, init_sv, params=params)
flags = {
    'exact': False,
    'train': True
}
params = {
    'n_neighbors': 4,
    'simi_type': 'ed',
    'f_shap': 'n*n',
    'rela': ['poly', 1],
    'train_idxs': [3, 11, 16],
    'm': (len(y_train) - 1) * plus_time
}
heur_shap.prepare(flags=flags, params=params)
heur_knn_plus_sv_18 = heur_shap.del_multi_points(del_idxs)
save_npy('heur_knn_plus_sv_18.npy', heur_knn_plus_sv_18)

# MC

mc_sv_18 = dynashap.mc_shap(deleted_x_train, deleted_y_train, x_test, y_test, model,
                            len(deleted_y_train) * normal_time)
save_npy('mc_sv_18.npy', mc_sv_18)


# ### 4.3 YN Shap Check
# 
# > This case is just used to proof that YNN-NNN is `ZERO ERROR`, which means that YN-NN algorithm will
# > not bring new error into the Shapley value.

# In[ ]:


yn_shap = dynashap.YnShap(x_train, y_train, x_test, y_test,
                          model, init_sv)
flags = {'exact': True}
yn_shap.prepare(2, flags)
yn_sv_18 = yn_shap.del_multi_points(del_idxs, flags)
save_npy('yn_sv_18.npy', yn_sv_18)

exact_sv_18 = dynashap.exact_shap(deleted_x_train, deleted_y_train, x_test, y_test, model)
save_npy('exact_sv_18.npy', exact_sv_18)


# In[ ]:


print('The variance between yn_sv and exact_sv: \t %f' %
      variance(exact_sv_18, normalize(exact_sv_18, yn_sv_18)))


# ### 4.4 Comparison

# In[ ]:


# Load
mc_plus_sv_18  = load_npy('mc_plus_sv_18.npy')

delta_sv_18    = load_npy('delta_sv_18.npy')
knn_sv_18      = load_npy('heur_knn_sv_18.npy')
knn_plus_sv_18 = load_npy('heur_knn_plus_sv_18.npy')
mc_sv_18       = load_npy('mc_sv_18.npy')


# In[ ]:


# Variance
comp(mc_plus_sv_18, delta_sv_18, 'delta')
comp(mc_plus_sv_18, mc_sv_18, 'mc')
comp(mc_plus_sv_18, knn_sv_18, 'knn')
comp(mc_plus_sv_18, knn_plus_sv_18, 'knn+')

