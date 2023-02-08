# -*- coding: utf-8 -*-
# Copyright (c) Haocheng Xia.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest
import random
from sklearn import svm

from dynashap.dynamic import (
    BaseShap, DeltaShap, PivotShap, HeurShap, YnShap
)


class TestApi(object):
    def setup(self):
        X = [[0.1, 0.2]] * 20
        y = ([0] * 10) + ([1] * 10)
        self.X = np.asarray(X)
        self.y = np.asarray(y)

        # Perturb
        for i in range(len(self.y)):
            self.X[i][0] += random.randint(0, 10) * 0.01
            self.X[i][1] += random.randint(0, 10) * 0.01

        self.model = svm.SVC(decision_function_shape='ovo')
        self.init_sv = np.zeros(20)

    def test_base_shap(self):
        base_shap = BaseShap(self.X, self.y, self.X, self.y,
                             self.model, self.init_sv)
        sv = base_shap.add_single_point(self.X[0], self.y[0],
                                        params={'method': 'avg'})
        assert len(sv) == len(self.init_sv) + 1
        sv = base_shap.add_single_point(self.X[0], self.y[0],
                                        params={'method': 'loo'})
        assert len(sv) == len(self.init_sv) + 1
        sv = base_shap.add_multi_points(self.X[:2], self.y[:2],
                                        params={'method': 'avg'})
        assert len(sv) == len(self.init_sv) + 2
        sv = base_shap.add_multi_points(self.X[:2], self.y[:2],
                                        params={'method': 'loo'})
        assert len(sv) == len(self.init_sv) + 2

    def test_delta_shap(self):
        delta_shap = DeltaShap(self.X, self.y, self.X, self.y,
                               self.model, self.init_sv)
        sv = delta_shap.add_single_point(self.X[0], self.y[0], 10)
        assert len(sv) == len(self.init_sv) + 1

    def test_pivot_shap(self):
        pivot_shap = PivotShap(self.X, self.y, self.X, self.y,
                               self.model, None)
        pivot_shap.prepare(100)
        sv = pivot_shap.add_single_point(self.X[0], self.y[0], proc_num=1, params={'method': 's'},
                                         flags={'flag_lsv': True})
        assert len(sv) == len(self.init_sv) + 1
        assert len(pivot_shap.lsv) == len(self.init_sv) + 1

    def test_heur_shap(self):
        heur_shap = HeurShap(self.X, self.y, self.X, self.y,
                             self.model, self.init_sv,
                             params={'method': 'knn'})
        heur_shap.prepare(params={'n_neighbors': 5})
        sv = heur_shap.add_multi_points(self.X[:2], self.y[:2])
        assert len(sv) == len(self.init_sv) + 2

        heur_shap = HeurShap(self.X, self.y, self.X, self.y,
                             self.model, self.init_sv,
                             params={'method': 'knn+'})
        heur_shap.prepare(flags={'exact': False,
                                 'train': True},
                          params={
                              'n_neighbors': 3,
                              'simi_type': 'ed',
                              'f_shap': 'n*n',
                              'rela': ['poly', 2],
                              'train_idxs': [3, 11],
                              'm': 10}
                          )
        sv = heur_shap.add_multi_points(self.X[:2], self.y[:2])
        assert len(sv) == len(self.init_sv) + 2

    def test_yn_shap(self):
        yn_shap = YnShap(self.X, self.y, self.X, self.y,
                         self.model, self.init_sv)
        yn_shap.prepare(1, flags={'exact': False},
                        params={'mc_type': 0, 'm': 2})
        sv = yn_shap.del_single_point(0)
        assert len(sv) == len(self.init_sv) - 1

        yn_shap = YnShap(self.X[:2], self.y[:2], self.X, self.y,
                         self.model, self.init_sv[:2])
        yn_shap.prepare(1, flags={'exact': True})
        sv = yn_shap.del_single_point(0)
        assert len(sv) == len(self.init_sv[:2]) - 1


if __name__ == '__main__':
    pytest.main("-s test_api.py")
