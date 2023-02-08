# -*- coding: utf-8 -*-
# Copyright (c) Haocheng Xia.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
dynashap.utils
~~~~~~~~~~~~~~
This module provides utility functions that are used within DynaShap.
"""

import time

from itertools import chain, combinations
import numpy as np
from sklearn import metrics
from typing import Iterator

from .exceptions import ParamError


def eval_utility(x_train, y_train, x_test, y_test, model) -> float:
    """Evaluate the coalition utility.
    """

    single_pred_label = (True if len(np.unique(y_train)) == 1
                         else False)

    if single_pred_label:
        y_pred = [y_train[0]] * len(y_test)
    else:
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

    return metrics.accuracy_score(y_test, y_pred, normalize=True)


def power_set(iterable) -> Iterator:
    """Generate the power set of the all elements of an iterable obj.
    """
    s = list(iterable)
    return chain.from_iterable(combinations(s, r)
                               for r in range(1, len(s) + 1))


def time_function(f, *args) -> float:
    """Call a function f with args and return the time (in seconds) 
    that it took to execute.
    """

    tic = time.time()
    f(*args)
    toc = time.time()
    return toc - tic


def get_ele_idxs(ele, ele_list) -> list:
    """Return all index of a specific element in the element list
    """
    idx = -1
    if not isinstance(ele_list, list):
        ele_list = list(ele_list)
    n = ele_list.count(ele)
    idxs = [0] * n
    for i in range(n):
        idx = ele_list.index(ele, idx + 1, len(ele_list))
        idxs[i] = idx
    return idxs


def eval_simi(x, y, idx, simi_type='ed') -> np.ndarray:
    """Return similarity value change array

    :param x:   features
    :param y:   labels
    :param idx: index
    :param simi_type:
        `ed`  - reciprocal of Euclid Distance
        `cos` - cosine distance
    """

    n = len(y)
    simi = np.zeros(n)
    if simi_type == 'ed':
        for i in range(n):
            if sum(x[idx] - x[i]) == 0:
                simi[i] = 0
            else:
                simi[i] = 1 / np.sqrt(sum((x[idx] - x[i]) ** 2))
    elif simi_type == 'cos':
        for i in range(n):
            if i == idx:
                pass
            else:
                simi[i] = np.dot(x[idx], x[i]) / (np.linalg.norm(x[idx]) *
                                                  np.linalg.norm(x[i]))
    else:
        raise ParamError('invalid simi_type')
    return simi


def eval_svc(new_sv, origin_sv) -> np.ndarray:
    """Return Shapley value change array
    """
    return new_sv - origin_sv


def split_permutation_num(m, num) -> np.ndarray:
    """Split a number into num numbers

    e.g. split_permutations(9, 2) -> [4, 5]
         split_permutations(9, 3) -> [3, 3, 3]

    :param m: the original num
    :param num: split into num numbers
    :return: np.ndarray
    """

    assert m > 0
    quotient = int(m / num)
    remainder = m % num
    if remainder > 0:
        perm_arr = [quotient] * (num - remainder) + [quotient + 1] * remainder
    else:
        perm_arr = [quotient] * num
    return np.asarray(perm_arr)


def split_permutations_t_list(permutations, t_list, num) -> list:
    """Split permutations and t_list

    :param permutations: the original num
    :param t_list: the t list
    :param num: split into num numbers
    :return: list
    """

    m = len(permutations)
    m_list = split_permutation_num(m, num)
    res = list()
    for local_m in m_list:
        res.append([permutations[:local_m], t_list[:local_m]])
        permutations = permutations[local_m:]
        t_list = t_list[local_m:]
    return res
