# -*- coding: utf-8 -*-
# Copyright (c) Haocheng Xia.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest
from sklearn import svm

from dynashap.utils import (
    eval_utility, power_set, get_ele_idxs
)


class TestUtils(object):
    def setup(self):
        X = [[0.11, 0.22],
             [0.31, 0.12],
             [0.12, 0.25],
             [0.33, 0.14]]
        y = [0, 1, 0, 1]
        self.X = np.asarray(X)
        self.y = np.asarray(y)
        self.model = svm.SVC(decision_function_shape='ovo')

        self.ele_list = [0, 1, 2]
        self.power_set = [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]

        self.name_list = ['a', 'b', 'a', 'b', 'a', 'b']
        self.a_idx = [0, 2, 4]

    def test_eval_utility(self):
        u = eval_utility(self.X[:1], self.y[:1],
                         self.X[2:], self.y[2:],
                         self.model)
        assert u == 0.5

    def test_power_set(self):
        assert list(power_set(self.ele_list)) == self.power_set

    def test_get_ele_idxs(self):
        assert get_ele_idxs('a', self.name_list) == self.a_idx


if __name__ == '__main__':

    pytest.main("-s test_utils.py")