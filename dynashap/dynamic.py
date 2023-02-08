# -*- coding: utf-8 -*-
# Copyright (c) Haocheng Xia.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
dynashap.dynamic
~~~~~~~~~~~~~~~~

This module provides the implementation of various algorithms for calculating
dynamic Shapley value.
"""

from functools import partial
from itertools import product
from multiprocessing import Pool
import numpy as np
from sklearn import neighbors
from tqdm import tqdm, trange

from .exceptions import (
    UnImpException, ParamError
)
from .structures import SimiPreData
from .utils import (
    eval_utility, eval_simi, eval_svc, power_set, get_ele_idxs,
    split_permutations_t_list, split_permutation_num
)


class DynaShap(object):
    """A base class for dynamic Shapley value computation.
    """

    def __init__(self, x_train, y_train, x_test, y_test,
                 model, init_sv, flags=None, params=None) -> None:
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        self.model = model
        self.init_sv = init_sv

        self.flags = flags
        self.params = params

    def add_single_point(self, add_point_x, add_point_y,
                         flags=None, params=None) -> np.ndarray:
        raise UnImpException('add_single_point')

    def del_single_point(self, del_point_idx,
                         flags=None, params=None) -> np.ndarray:
        raise UnImpException('del_single_point')

    def add_multi_points(self, add_points_x, add_points_y,
                         flags=None, params=None) -> np.ndarray:
        raise UnImpException('add_multi_points')

    def del_multi_points(self, del_points_idx,
                         flags=None, params=None) -> np.ndarray:
        raise UnImpException('del_multi_points')


class BaseShap(DynaShap):
    """Baseline algorithms for dynamically add point(s).
    """

    def __init__(self, x_train, y_train, x_test, y_test,
                 model, init_sv) -> None:
        super().__init__(x_train, y_train, x_test, y_test,
                         model, init_sv)

    def add_single_point(self, add_point_x, add_point_y,
                         flags=None, params=None) -> np.ndarray:
        """
        Add a single point and update the Shapley value with
        baseline algorithm. (Avg & LOO)

        :param np.ndarray add_point_x:  the features of the adding point
        :param np.ndarray add_point_y:  the label of the adding point
        :param dict flags:              (unused yet)
        :param dict params:             {'method': 'avg' or 'loo'}
        :return: Shapley value array `base_sv`
        :rtype: numpy.ndarray
        """
        if params is None:
            params = {'method': 'avg'}

        return self.add_multi_points(np.asarray([add_point_x]),
                                     np.asarray([add_point_y]),
                                     None, params)

    def add_multi_points(self, add_points_x, add_points_y,
                         flags=None, params=None) -> np.ndarray:
        """
        Add multiple points and update the Shapley value with
        baseline algorithm. (Avg & LOO)

        :param np.ndarray add_points_x:  the features of the adding points
        :param np.ndarray add_points_y:  the labels of the adding points
        :param None flags:               (unused yet)
        :param dict params:              {'method': 'avg' or 'loo'}
        :return: Shapley value array `base_sv`
        :rtype: numpy.ndarray
        """

        if params is None:
            params = {'method': 'avg'}
        # select 'avg' or 'loo'
        method = params['method']

        add_num = len(add_points_y)

        if method == 'avg':
            nsv = np.sum(self.init_sv) / len(self.init_sv)
            return np.append(self.init_sv, [nsv] * add_num)

        elif method == 'loo':
            n = len(self.init_sv)
            loo_vals = np.zeros(n + add_num)

            new_x_train = np.append(self.x_train, add_points_x, axis=0)
            new_y_train = np.append(self.y_train, add_points_y)

            for i in range(n + add_num):
                idxs = np.delete(np.arange(n + add_num), i)

                # Evaluate utility excluding i
                temp_x, temp_y = new_x_train[idxs], new_y_train[idxs]
                u_no_i = eval_utility(temp_x, temp_y, self.x_test,
                                      self.y_test, self.model)

                # Evaluate utility including i
                idxs = np.append(idxs, i)
                temp_x, temp_y = new_x_train[idxs], new_y_train[idxs]
                u_with_i = eval_utility(temp_x, temp_y, self.x_test,
                                        self.y_test, self.model)

                loo_vals[i] = u_with_i - u_no_i

            base_sv = self.init_sv
            for i in range(add_num):
                loo_order = np.argsort(np.append(loo_vals[:n],
                                                 loo_vals[n + i]))
                idx = np.where(loo_order == n)[0]
                if 0 < idx < n:
                    nsv = (self.init_sv[loo_order[idx - 1]] +
                           self.init_sv[loo_order[idx + 1]]) / 2
                    base_sv = np.append(base_sv, nsv)
                elif idx == 0:
                    base_sv = np.append(base_sv, self.init_sv[loo_order[1]])
                else:
                    base_sv = np.append(base_sv, self.init_sv[loo_order[-2]])
            return base_sv

        else:
            raise ParamError


class PivotShap(DynaShap):
    """Pivot-based algorithm with the same sampled permutations
    """

    def __init__(self, x_train, y_train, x_test, y_test,
                 model, init_sv) -> None:
        super().__init__(x_train, y_train, x_test, y_test,
                         model, init_sv)
        # If no prepare, pass left sv by init sv
        self.lsv = init_sv
        self.t_list = None
        self.permutations = None
        self.proc_num = None

    def prepare(self, m, proc_num=1) -> np.ndarray:
        """
        Prepare procedure needed by pivot based dynamic Shapley algorithm.
        (Phase Initialization)

        Calculating the left part of the permutations.

        :param proc_num: (optional) Assign the proc num with multi-processing
                         support. Defaults to ``1``.
        :param int m:    The number of the permutations.
        :return: the lsv `left_sv` and the t list
        :rtype: np.ndarray
        """

        if proc_num <= 0:
            raise ValueError('Invalid proc num.')

        self.proc_num = proc_num

        n = len(self.y_train)
        self.init_sv = np.zeros(n)

        args = split_permutation_num(m, proc_num)
        pool = Pool()
        func = partial(PivotShap._pivot_prepare_sub_task,
                       self.x_train, self.y_train,
                       self.x_test, self.y_test, self.model)
        ret = pool.map(func, args)
        pool.close()
        pool.join()
        ret_arr = np.asarray(ret, dtype=object)

        self.lsv = np.sum([r[0] for r in ret_arr], axis=0) / m
        self.permutations = np.concatenate([r[1] for r in ret_arr], axis=0)
        self.t_list = np.concatenate([r[2] for r in ret_arr], axis=0)
        return self.lsv

    @staticmethod
    def _pivot_prepare_sub_task(x_train, y_train, x_test, y_test, model,
                                local_m) -> (np.ndarray, np.ndarray, list):
        local_state = np.random.RandomState(None)

        n = len(y_train)
        lsv = np.zeros(n)
        idxs = np.arange(n)
        t_list = list()
        permutations = list()

        for _ in trange(local_m):
            local_state.shuffle(idxs)
            # Draw t from 0 to n
            t = local_state.randint(0, n + 1)

            # Record trim position and the corresponding sequence
            permutations.append(idxs.tolist())
            t_list.append(t)

            old_u = 0
            # TODO: Decide whether calculate sv here
            for j in range(1, t + 1):
                temp_x, temp_y = x_train[idxs[:j]], y_train[idxs[:j]]
                temp_u = eval_utility(temp_x, temp_y, x_test,
                                      y_test, model)
                lsv[idxs[j - 1]] += temp_u - old_u
                old_u = temp_u
        return lsv, permutations, t_list

    def add_single_point(self, add_point_x, add_point_y, m=None, proc_num=1,
                         flags=None, params=None) -> np.ndarray:
        """
        Add a single point and update the Shapley value with pivot based
        algorithm.

        :param np.ndarray add_point_x:  the features of the adding point
        :param np.ndarray add_point_y:  the label of the adding point
        :param int m:                   the num of permutations
        :param int proc_num:            (optional) Assign the proc num with
                                        multi-processing support. Defaults
                                        to ``1``.
        :param dict flags:              {'flag_lsv': False} ('flag_lsv'
                                        represents that update left sv or not
                                        Defaults to ``False``.)
        :param dict params:             {'method': 'd' or 's'} (with the
                                        different or same permutations)
                                        Defaults to
        :return: Shapley value array `pivot_sv`
        :rtype: numpy.ndarray
        """

        if flags is None:
            flags = {'flag_lsv': False}
        if params is None:
            params = {'method': 'd'}

        # Extract flags & params
        flag_lsv = flags['flag_lsv']
        method = params['method']

        new_x_train = np.append(self.x_train, [add_point_x], axis=0)
        new_y_train = np.append(self.y_train, add_point_y)

        # Init left part and right part
        lsv = np.append(self.lsv, 0)

        pool = Pool()
        if method == 's':
            # With the same permutations
            m = len(self.t_list)
            args = split_permutations_t_list(self.permutations,
                                             self.t_list, proc_num)
            f = PivotShap._pivot_add_sub_task_s
        else:
            # With different permutations
            if m is None:
                m = len(self.t_list)
            args = split_permutation_num(m, proc_num)
            f = PivotShap._pivot_add_sub_task_d

        func = partial(f, new_x_train, new_y_train,
                       self.x_test, self.y_test, self.model)
        ret = pool.map(func, args)
        pool.close()
        pool.join()
        ret_arr = np.asarray(ret, dtype=object)
        rsv = np.sum([r[0] for r in ret_arr], axis=0) / m
        delta_lsv = np.sum([r[1] for r in ret_arr], axis=0) / m
        pivot_sv = lsv + rsv

        if flag_lsv:
            self.x_train = new_x_train
            self.y_train = new_y_train
            self.lsv = lsv * 2 / 3 + delta_lsv
            if method == 's':
                self.permutations = np.concatenate([r[2] for r in ret_arr],
                                                   axis=0)
                self.t_list = np.concatenate([r[3] for r in ret_arr], axis=0)
        return np.asarray(pivot_sv, dtype=float)

    @staticmethod
    def _pivot_add_sub_task_s(new_x_train, new_y_train, x_test, y_test,
                              model, local_permutations_t_list) -> \
        (np.ndarray, np.ndarray, np.ndarray, list):
        """The Shapley value calculation with same permutations.
        """

        local_state = np.random.RandomState(None)

        n = len(new_y_train) - 1
        rsv = np.zeros(n + 1)
        delta_lsv = np.zeros(n + 1)

        new_permutations = list()
        p_list = list()
        # Extract local_permutations and local t list
        local_permutations = local_permutations_t_list[0]
        local_t_list = local_permutations_t_list[1]

        # Using t list update the permutation
        for permutation, t in tqdm(zip(local_permutations, local_t_list),
                                   total=len(local_t_list)):
            # Insert the index of added point into position t
            permutation = np.insert(permutation, t, n)
            # Draw p from 0 to n + 1 (the size of new_y_train)
            p = local_state.randint(0, n + 2)
            new_permutations.append(permutation.tolist())
            p_list.append(p)

            if t == 0:
                old_u = 0
            else:
                temp_x, temp_y = new_x_train[permutation[:t]], \
                                 new_y_train[permutation[:t]]
                old_u = eval_utility(temp_x, temp_y, x_test, y_test, model)
            for i in range(t + 1, n + 2):
                temp_x, temp_y = new_x_train[permutation[:i]], \
                                 new_y_train[permutation[:i]]
                temp_u = eval_utility(temp_x, temp_y, x_test, y_test, model)

                if p >= i:
                    delta_lsv[permutation[i - 1]] += temp_u - old_u
                rsv[permutation[i - 1]] += temp_u - old_u
                old_u = temp_u

        return rsv, delta_lsv, new_permutations, p_list

    @staticmethod
    def _pivot_add_sub_task_d(new_x_train, new_y_train, x_test, y_test, model,
                              local_m) -> (np.ndarray, np.ndarray):
        """The Shapley value calculation with different permutations.
        """

        local_state = np.random.RandomState(None)

        n = len(new_y_train) - 1
        rsv = np.zeros(n + 1)
        delta_lsv = np.zeros(n + 1)
        idxs = np.arange(n + 1)

        for _ in trange(local_m):
            # Draw p from 0 to n + 1 (the size of new_y_train)
            p = local_state.randint(0, n + 2)
            local_state.shuffle(idxs)
            t = 0
            for t in range(n + 1):
                # Find the new added point's idx
                if idxs[t] == n:
                    break

            # Evaluate utility excluding the new added point
            if t == 0:
                old_u = 0
            else:
                temp_x, temp_y = new_x_train[idxs[:t]], new_y_train[idxs[:t]]
                old_u = eval_utility(temp_x, temp_y, x_test, y_test, model)

            # Evaluate utility including the new added point (from t+1 to n+1)
            for j in range(t + 1, n + 2):
                temp_x, temp_y = new_x_train[idxs[:j]], new_y_train[idxs[:j]]
                temp_u = eval_utility(temp_x, temp_y, x_test, y_test, model)

                if p >= j:
                    delta_lsv[idxs[j - 1]] += (temp_u - old_u)

                rsv[idxs[j - 1]] += temp_u - old_u
                old_u = temp_u

        return rsv, delta_lsv


class DeltaShap(DynaShap):
    """Delta based algorithm for dynamically add/delete a single point.
    """

    def __init__(self, x_train, y_train, x_test, y_test,
                 model, init_sv) -> None:
        super().__init__(x_train, y_train, x_test, y_test,
                         model, init_sv)
        self.m = None

    def add_single_point(self, add_point_x, add_point_y, m, proc_num=1,
                         flags=None, params=None) -> np.ndarray:
        """
        Add a single point and update the Shapley value with delta based
        algorithm.

        :param np.ndarray add_point_x:  the features of the adding point
        :param np.ndarray add_point_y:  the label of the adding point
        :param int m:                   the number of permutations
        :param int proc_num:            the number of proc
        :param dict flags:              (optional) {'flag_update': True or
                                        False} Defaults to ``False``.
        :param dict params:             (unused yet)
        :return: Shapley value array `delta_sv`
        :rtype: numpy.ndarray
        """

        self.m = m
        if proc_num <= 0:
            raise ValueError('Invalid proc num.')

        if flags is None:
            flags = {'flag_update': False}

        flag_update = flags['flag_update']

        # assign the permutation of each process
        args = split_permutation_num(m, proc_num)
        pool = Pool()
        func = partial(DeltaShap._delta_add_sub_task,
                       self.x_train, self.y_train,
                       self.x_test, self.y_test,
                       self.model, add_point_x, add_point_y)
        ret = pool.map(func, args)
        pool.close()
        pool.join()
        ret_arr = np.asarray(ret)
        delta = np.sum(ret_arr, axis=0) / m
        delta_sv = np.append(self.init_sv, 0) + delta

        if flag_update:
            self.x_train = np.append(self.x_train, [add_point_x], axis=0)
            self.y_train = np.append(self.y_train, add_point_y)
            self.init_sv = delta_sv

        return delta_sv

    @staticmethod
    def _delta_add_sub_task(x_train, y_train, x_test, y_test, model,
                            add_point_x, add_point_y, local_m) -> np.ndarray:
        local_state = np.random.RandomState(None)

        n = len(y_train)
        idxs = np.arange(n)
        delta = np.zeros(n + 1)

        origin_margin = eval_utility([add_point_x], [add_point_y],
                                     x_test, y_test, model)

        delta[-1] += origin_margin / (n + 1) * local_m

        for _ in trange(local_m):
            local_state.shuffle(idxs)
            for i in range(1, n + 1):
                temp_x, temp_y = (x_train[idxs[:i]],
                                  y_train[idxs[:i]])

                u_no_np = eval_utility(temp_x, temp_y, x_test, y_test,
                                       model)

                u_with_np = eval_utility(np.append(temp_x, [add_point_x],
                                                   axis=0),
                                         np.append(temp_y, add_point_y),
                                         x_test, y_test, model)

                current_margin = u_with_np - u_no_np

                delta[idxs[i - 1]] += ((current_margin - origin_margin)
                                       / (n + 1) * i)

                delta[-1] += current_margin / (n + 1)
                origin_margin = current_margin
        return delta

    def del_single_point(self, del_point_idx, m, proc_num=1,
                         flags=None, params=None) -> np.ndarray:
        """
        Delete a single point and update the Shapley value with
        delta based algorithm. (KNN & KNN+)

        :param int del_point_idx:   the index of the deleting point
        :param m:                   the number of permutations
        :param proc_num:            the number of proc
        :param dict flags:          (optional) {'flag_update': True or False},
                                    Defaults to ``False``.
        :param dict params:         (unused yet)
        :return: Shapley value array `delta_sv`
        :rtype: numpy.ndarray
        """

        self.m = m

        if proc_num <= 0:
            raise ValueError('Invalid proc num.')

        if flags is None:
            flags = {'flag_update': False}

        flag_update = flags['flag_update']

        # assign the permutation of each process
        args = split_permutation_num(m, proc_num)
        pool = Pool()
        func = partial(DeltaShap._delta_del_sub_task,
                       self.x_train, self.y_train,
                       self.x_test, self.y_test,
                       self.model, del_point_idx)
        ret = pool.map(func, args)
        pool.close()
        pool.join()
        ret_arr = np.asarray(ret)
        delta = np.sum(ret_arr, axis=0) / m
        delta_sv = np.delete(self.init_sv, del_point_idx) + delta

        if flag_update:
            self.x_train = np.delete(self.x_train, del_point_idx, axis=0)
            self.y_train = np.delete(self.y_train, del_point_idx)
            self.init_sv = delta_sv

        return delta_sv

    @staticmethod
    def _delta_del_sub_task(x_train, y_train, x_test, y_test,
                            model, del_point_idx, local_m) -> np.ndarray:
        local_state = np.random.RandomState(None)

        n = len(y_train)
        deleted_idxs = np.delete(np.arange(n), del_point_idx)
        fixed_idxs = np.copy(deleted_idxs)
        delta = np.zeros(n - 1)

        origin_margin = eval_utility([x_train[del_point_idx, :]],
                                     [y_train[del_point_idx]],
                                     x_test, y_test, model)

        for _ in trange(local_m):
            local_state.shuffle(deleted_idxs)
            for j in range(1, n):
                temp_x, temp_y = (x_train[deleted_idxs[:j]],
                                  y_train[deleted_idxs[:j]])

                acc_no_op = eval_utility(temp_x, temp_y, x_test,
                                         y_test, model)

                temp_x, temp_y = (np.append(temp_x, [x_train[del_point_idx]],
                                            axis=0),
                                  np.append(temp_y, y_train[del_point_idx]))

                acc_with_op = eval_utility(temp_x, temp_y, x_test,
                                           y_test, model)

                current_margin = acc_with_op - acc_no_op

                idx = np.where(fixed_idxs == deleted_idxs[j - 1])[0]
                delta[idx] += ((-current_margin + origin_margin)
                               / n * j)
                origin_margin = current_margin
        return delta


class YnShap(DynaShap):
    """YN-NN algorithm for dynamically delete point(s).
    """

    def __init__(self, x_train, y_train, x_test, y_test,
                 model, init_sv) -> None:
        super().__init__(x_train, y_train, x_test, y_test,
                         model, init_sv)
        self.MAX_DEL_NUM = 2
        self.del_num = None
        self.yn = None
        self.nn = None

    def prepare(self, del_num, flags=None,
                params=None):
        """
        The prepare procedure needed by YN-NN algorithm, which needs
        to fill in the multi-dimension array.

        :param del_num:        the number of points which need to be deleted
        :param dict flags:     {'exact': True or False,}
        :param dict params:    (optional) {'mc_type': 0 or 1, 'm': ...} (it is
                               needed when 'exact' is False)
        :return: `yn` and `nn`
        :rtype: tuple([np.ndarray, np.ndarray]
        """
        if flags is None:
            flags = {'exact': True}

        n = len(self.y_train)
        self.del_num = del_num

        if self.del_num > self.MAX_DEL_NUM:
            raise ParamError('the number of delete points cannot > %d'
                             % self.MAX_DEL_NUM)

        shape = tuple([n]) * (del_num + 2)
        self.yn = np.zeros(shape=shape)
        self.nn = np.zeros(shape=shape)

        if flags['exact']:
            fact = np.math.factorial
            coef = np.zeros(n)
            coalition = np.arange(n)

            for s in range(n):
                coef[s] = fact(s) * fact(n - s - 1) / fact(n)

            sets = list(power_set(coalition))
            for i in trange(len(sets)):
                temp_x, temp_y = (self.x_train[list(sets[i])],
                                  self.y_train[list(sets[i])])
                u = eval_utility(temp_x, temp_y, self.x_test, self.y_test,
                                 self.model)

                # Assign utility to array
                l = len(sets[i])
                Y = list(sets[i])
                N = list(set(coalition) - set(Y))

                if self.del_num == 1:
                    for j, k in product(Y, N):
                        self.yn[j, k, l] += u * coef[l - 1]
                    for j, k in product(N, N):
                        self.nn[j, k, l] += u * coef[l]
                else:
                    for j, k, p in product(Y, N, N):
                        self.yn[j, k, p, l] += u * coef[l - 1]
                    for j, k, p in product(N, N, N):
                        self.nn[j, k, p, l] += u * coef[l]

        else:
            mc_type = params['mc_type']
            m = params['m']

            idxs = np.arange(n)
            for _ in trange(int(n * m)):
                np.random.shuffle(idxs)
                old_u = 0
                for l in range(1, n + 1):
                    temp_x, temp_y = (self.x_train[idxs[:l]],
                                      self.y_train[idxs[:l]])
                    temp_u = eval_utility(temp_x, temp_y, self.x_test,
                                          self.y_test, self.model)

                    # Assign utility to array
                    N = idxs[l:]
                    if mc_type == 0:
                        j = idxs[l - 1]
                        if self.del_num == 1:
                            for k in N:
                                self.yn[j, k, l] += temp_u
                                self.nn[j, k, l - 1] += old_u
                        else:
                            for k, p in product(N, N):
                                self.yn[j, k, p, l] += temp_u
                                self.nn[j, k, p, l - 1] += old_u
                    else:
                        Y = idxs[:l]
                        if self.del_num == 1:
                            for j, k in product(Y, N):
                                self.yn[j, k, l] += temp_u
                            for j, k in product(N, N):
                                self.nn[j, k, l] += temp_u
                        else:
                            for j, k, p in product(Y, N, N):
                                self.yn[j, k, p, l] += temp_u
                            for j, k, p in product(N, N, N):
                                self.nn[j, k, p, l] += temp_u

                    old_u = temp_u

            self.yn, self.nn = self.yn / m, self.nn / m

        return self.yn, self.nn

    def del_single_point(self, del_point_idx,
                         flags=None, params=None) -> np.ndarray:
        """
        Delete a single point and update the Shapley value with
        YN-NN algorithm.

        :param int del_point_idx:    the index of the deleting point
        :param dict flags:           {'exact': True or False}
        :param dict params:          (unused yet)
        :return: Shapley value array `yn_sv`
        :rtype: numpy.ndarray
        """

        if flags is None:
            flags = {'exact': True}

        return self.del_multi_points([del_point_idx], flags, params)

    def del_multi_points(self, del_points_idx,
                         flags=None, params=None) -> np.ndarray:
        """
        Delete multiple points and update the Shapley value with
        YN-NN (YNN-NNN) algorithm. (KNN & KNN+)

        :param list del_points_idx:  the index of the deleting points
        :param dict flags:           {'exact': True or False}
        :param dict params:          (unused yet)
        :return: Shapley value array `yn_sv`
        :rtype: numpy.ndarray
        """

        if flags is None:
            flags = {'exact': True}

        if len(del_points_idx) > self.del_num:
            raise ParamError('delete too many points')

        n = len(self.y_train)
        yn_sv = np.zeros(n)
        walk_arr = np.delete(np.arange(n), np.asarray(del_points_idx))

        for i, j in product(walk_arr, range(1, 1 + len(walk_arr))):
            t = tuple(del_points_idx[:self.del_num])
            if not flags['exact'] and params['mc_type'] == 0:
                modi_coef_method = 0
            else:
                modi_coef_method = 1
            yn_sv[i] += ((self.yn[(i,) + t + (j,)] -
                          self.nn[(i,) + t + (j - 1,)]) *
                         YnShap._modi_coef(n, j, self.del_num,
                                           modi_coef_method))
        return np.delete(yn_sv, del_points_idx)

    @staticmethod
    def _modi_coef(n, j, num, method):
        res = 1
        for i in range(num):
            if method == 0:
                res *= n - 1 - i
            else:
                res *= n - i
            res /= n - j - i
        return res


class HeurShap(DynaShap):
    """Heuristic dynamic Shapley algorithm, including KNN and KNN+ version
    """

    def __init__(self, x_train, y_train, x_test, y_test,
                 model, init_sv, flags=None, params=None) -> None:
        """
        :param flags: unused yet
        :param params: {'method': 'knn' or 'knn+'}
        """
        super().__init__(x_train, y_train, x_test, y_test,
                         model, init_sv)

        if params is None:
            params = {'method': 'knn'}

        # Extract param
        self.method = params['method']

        self.n_neighbors = None
        self.clf = None
        self.simi_type = None
        self.m = None
        self.spd = None

    def prepare(self, flags=None,
                params=None) -> None:
        """
        The prepare procedure needed by heuristic algorithm, including
        KNN clf training, curve functions generating and etc.

        :param dict flags:  {'exact': True or False,
                             'train': True or False}
        :param dict params: {'n_neighbors': 5,
                             'simi_type': 'ed' or 'cos',
                             'f_shap': 'n*n',
                             'rela': ['poly', 2],
                             'train_idxs': []}
                             (['poly', x] | x in [1, ..., N],
                             in default x is 2)
        :return: None
        :rtype: None
        """

        if flags is None:
            flags = {'exact': False, 'train': True}

        # Extract param & flags
        self.n_neighbors = params['n_neighbors']

        self.clf = (neighbors.NearestNeighbors(n_neighbors=self.n_neighbors).
                    fit(self.x_train, self.y_train))
        if self.method == 'knn+':
            # Curve fitting
            flag_train = flags['train']
            self.simi_type = params['simi_type']

            n = len(self.y_train)

            if not flag_train:
                self.spd = SimiPreData(params)
            else:
                flag_ext = flags['exact']
                train_idxs = params['train_idxs']
                if not flag_ext:
                    self.m = params['m']

                svs = np.zeros((len(train_idxs), len(self.y_train) - 1))

                for i, train_idx in enumerate(train_idxs):
                    idxs = np.delete(np.arange(n), train_idx)
                    if flag_ext:
                        svs[i] = exact_shap(self.x_train[idxs],
                                            self.y_train[idxs],
                                            self.x_test, self.y_test,
                                            self.model)
                    else:
                        svs[i] = mc_shap(self.x_train[idxs],
                                         self.y_train[idxs],
                                         self.x_test, self.y_test,
                                         self.model, self.m)
                # Fill in SimiPreData
                self.spd = SimiPreData({'train_idxs': train_idxs,
                                        'train_svs': svs})

            self._fit_curve(params)

    def add_single_point(self, add_point_x, add_point_y, flags=None,
                         params=None) -> np.ndarray:
        """
        Add a single point and update the Shapley value with
        heuristic algorithm. (KNN & KNN+)

        :param np.ndarray add_point_x:  the features of the adding point
        :param np.ndarray add_point_y:  the label of the adding point
        :param dict flags:              (unused yet)
        :param dict params:             (unused yet)
        :return: Shapley value array `knn_sv` or `knn_plus_sv`
        :rtype: numpy.ndarray
        """
        return self.add_multi_points(np.asarray([add_point_x]),
                                     np.asarray([add_point_y]),
                                     flags, params)

    def add_multi_points(self, add_points_x, add_points_y, flags=None,
                         params=None) -> np.ndarray:
        """
        Add multiple points and update the Shapley value with
        heuristic algorithm. (KNN & KNN+)

        :param np.ndarray add_points_x:  the features of the adding points
        :param np.ndarray add_points_y:  the labels of the adding points
        :param dict flags:               (unused yet)
        :param dict params:              (unused yet)
        :return: Shapley value array `knn_sv` or `knn_plus_sv`
        :rtype: numpy.ndarray
        """
        self.add_points_x = add_points_x
        self.add_points_y = add_points_y

        n = len(self.init_sv)
        add_num = len(add_points_y)
        knn_sv = np.append(self.init_sv, [0] * add_num)

        for i in range(add_num):
            x = add_points_x[i]
            neighbor_list = self.clf.kneighbors([x], self.n_neighbors,
                                                False)[0]
            nsv = (np.sum(self.init_sv[neighbor_list]) /
                   self.n_neighbors)
            knn_sv[i + n] = nsv
        if self.method == 'knn':
            return knn_sv
        else:
            # KNN+
            knn_plus_sv = knn_sv

            simi_type = self.simi_type
            svc = np.zeros(n)

            for r_idx in trange(add_num):
                x_train = np.append(self.x_train, [add_points_x[r_idx]],
                                    axis=0)
                y_train = np.append(self.y_train, add_points_y[r_idx])
                # Always add one point
                simi = eval_simi(x_train, y_train, n, simi_type)
                # Calculate svc with curve functions
                for i in range(n):
                    # Select the corresponding curve function
                    f = np.poly1d(self.curve_funcs
                                  [list(self.f_labels)
                                   .index(add_points_y[r_idx])]
                                  [list(self.all_labels)
                                   .index(self.y_train[i])])
                    svc[i] += -f(simi[i]) if simi[i] != 0 else 0

            added_x_train = np.append(self.x_train, add_points_x, axis=0)
            added_y_train = np.append(self.y_train, add_points_y)

            new_u = eval_utility(added_x_train, added_y_train,
                                 self.x_test, self.y_test,
                                 self.model)
            knn_plus_sv[:n] += svc
            knn_plus_sv *= new_u / np.sum(knn_plus_sv)
            return knn_plus_sv

    def del_single_point(self, del_point_idx, flags=None,
                         params=None) -> np.ndarray:
        """
        Delete a single point and update the Shapley value with
        heuristic algorithm. (KNN & KNN+)

        :param int del_point_idx:  the index of the deleting point
        :param dict flags:         (unused yet)
        :param dict params:        (unused yet)
        :return: Shapley value array `knn_sv` or `knn_plus_sv`
        :rtype: numpy.ndarray
        """
        return self.del_multi_points([del_point_idx], flags, params)

    def del_multi_points(self, del_points_idx, flags=None,
                         params=None) -> np.ndarray:
        """
        Delete multiple points and update the Shapley value with
        heuristic algorithm. (KNN & KNN+)

        :param list del_points_idx:  the index of the deleteing points
        :param dict flags:           (unused yet)
        :param dict params:          (unused yet)
        :return: Shapley value array `knn_sv` or `knn_plus_sv`
        :rtype: numpy.ndarray
        """
        self.del_points_idx = del_points_idx
        n = len(self.init_sv)

        knn_sv = np.delete(self.init_sv, del_points_idx)

        idxs = np.arange(n)
        deleted_idxs = np.delete(idxs, del_points_idx)
        deleted_x_train = self.x_train[deleted_idxs]
        deleted_y_train = self.y_train[deleted_idxs]
        # Update clf
        clf = (neighbors.NearestNeighbors(n_neighbors=self.n_neighbors)
               .fit(deleted_x_train, deleted_y_train))
        for i in del_points_idx:
            x = self.x_train[i]
            neighbor_list = clf.kneighbors([x], self.n_neighbors, False)[0]
            for k in neighbor_list:
                idx = deleted_idxs[k]
                knn_sv[k] += self.init_sv[idx] / self.n_neighbors
        if self.method == 'knn':
            return knn_sv
        else:
            # KNN+
            knn_plus_sv = knn_sv
            simi_type = self.simi_type

            svc = np.zeros(n)

            f_labels = list(self.f_labels)
            all_labels = list(self.all_labels)
            for idx in tqdm(del_points_idx):
                simi = eval_simi(self.x_train, self.y_train, idx, simi_type)
                # Calculate svc with curve functions
                for i in range(n):
                    # Select the corresponding curve function
                    f = np.poly1d(self.curve_funcs
                                  [f_labels.index(self.y_train[idx])]
                                  [all_labels.index(self.y_train[i])])
                    svc[i] += f(simi[i]) if simi[i] != 0 else 0

                new_u = eval_utility(deleted_x_train, deleted_y_train,
                                     self.x_test, self.y_test,
                                     self.model)
                knn_plus_sv += svc[deleted_idxs]
                knn_plus_sv *= new_u / np.sum(knn_plus_sv)
            return knn_plus_sv

    def _fit_curve(self, params=None) -> None:
        """
        Generate curve functions which represent the relationship between
        the change of Shapley value and the similarity.
        """

        if params is None:
            params = {'f_shap': 'n*n', 'rela': ['poly', 2],
                      'simi_type': 'ed'}

        # Extract params
        f_shap = params['f_shap']
        rela = params['rela']
        simi_type = params['simi_type']

        self.f_labels = set(self.y_train[self.spd.train_idxs])
        self.all_labels = set(self.y_train)

        if rela[0] == 'poly':
            curve_funcs = np.zeros((len(self.all_labels),
                                    len(self.f_labels),
                                    rela[1] + 1))
        else:
            raise ParamError("relationship excepting 'ploy' "
                             "is NOT supported yet")

        if f_shap == 'n*n':

            for _, train_idx in product(self.f_labels, self.spd.train_idxs):
                current_label_fidx = list(self.f_labels).index(self.y_train
                                                               [train_idx])
                for idx, k in enumerate(self.all_labels):
                    label_idxs = get_ele_idxs(k, self.y_train)
                    try:
                        label_idxs.remove(train_idx)
                    except ValueError:
                        pass
                    simi = eval_simi(self.x_train, self.y_train,
                                     train_idx, simi_type)
                    # Include the train idx point, del when check same
                    svs_idx = self.spd.train_idxs.index(train_idx)
                    origin_sv = np.delete(self.init_sv, train_idx)
                    svc = eval_svc(self.spd.svs[svs_idx], origin_sv)
                    svc = np.insert(svc, train_idx, 0)

                    if rela[0] == 'poly':
                        x = simi[np.where(simi != 0)]
                        y = svc[np.where(simi != 0)]
                        z = np.polyfit(x, y, rela[1])
                        p = np.poly1d(z)
                        curve_funcs[current_label_fidx, idx] = p
        self.curve_funcs = curve_funcs


def exact_shap(x_train, y_train, x_test, y_test, model):
    """
    Calculating the Shapley value of data points with exact method
    (Shapley value definition)

    :param x_train:  features of train dataset
    :param y_train:  labels of train dataset
    :param x_test:   features of test dataset
    :param y_test:   labels of test dataset
    :param model:    the selected model
    :return: Shapley value array `sv`
    :rtype: numpy.ndarray
    """

    n = len(y_train)
    ext_sv = np.zeros(n)
    coef = np.zeros(n)
    fact = np.math.factorial
    coalition = np.arange(n)
    for s in range(n):
        coef[s] = fact(s) * fact(n - s - 1) / fact(n)
    sets = list(power_set(coalition))
    for idx in trange(len(sets)):
        temp_x, temp_y = x_train[list(sets[idx])], y_train[list(sets[idx])]
        u = eval_utility(temp_x, temp_y, x_test, y_test,
                         model)
        for i in sets[idx]:
            ext_sv[i] += coef[len(sets[idx]) - 1] * u
        for i in set(coalition) - set(sets[idx]):
            ext_sv[i] -= coef[len(sets[idx])] * u
    return ext_sv


def mc_shap(x_train, y_train, x_test, y_test, model,
            m, proc_num=1, flag_abs=False) -> np.ndarray:
    """
    Calculating the Shapley value of data points with
    Monte Carlo Method (multi-process supported)

    :param x_train:  features of train dataset
    :param y_train:  labels of train dataset
    :param x_test:   features of test dataset
    :param y_test:   labels of test dataset
    :param model:    the selected model
    :param m:        the permutation number
    :param proc_num: (optional) Assign the proc num with multi-processing
                     support. Defaults to ``1``.
    :param flag_abs: (optional) Whether use the absolution marginal
                     contribution. Defaults to ``False``.
    :return: Shapley value array `sv`
    :rtype: numpy.ndarray
    """

    if proc_num < 0:
        raise ValueError('Invalid proc num.')

    # assign the permutation of each process
    args = split_permutation_num(m, proc_num)
    pool = Pool()
    func = partial(_mc_shap_sub_task, x_train, y_train, x_test,
                   y_test, model, flag_abs)
    ret = pool.map(func, args)
    pool.close()
    pool.join()
    ret_arr = np.asarray(ret)
    return np.sum(ret_arr, axis=0) / m


def _mc_shap_sub_task(x_train, y_train, x_test, y_test, model,
                      flag_abs, local_m) -> np.ndarray:
    local_state = np.random.RandomState(None)

    n = len(y_train)
    sv = np.zeros(n)
    idxs = np.arange(n)
    for _ in trange(local_m):
        local_state.shuffle(idxs)
        old_u = 0
        for j in range(1, n + 1):
            temp_x, temp_y = x_train[idxs[:j]], y_train[idxs[:j]]
            temp_u = eval_utility(temp_x, temp_y, x_test, y_test, model)
            contribution = ((temp_u - old_u) if not flag_abs
                            else abs(temp_u - old_u))
            sv[idxs[j - 1]] += contribution
            old_u = temp_u
    return sv
