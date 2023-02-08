===========
Quick Start
===========

Eager to get started? This page gives a good introduction in how to get started
with DynaShap.

.. contents:: On this page
   :local:

First, make sure that:

* DynaShap is :ref:`installed <install>`

Making a dynamic Shapley value computation with DynaShap is very simple.

Begin by importing the DynaShap module::

    >>> import dynashap

Add A Single Point
------------------

This method is support by these classes:

* BaseShap

    >>> base_shap = dynashap.BaseShap(x_train, y_train, x_test, y_test, model, init_sv)

    When you add a data point with BaseShap, you can choose update Shapley value with `average` method or `leave one out` method.

    * `average` method:

    >>> sv = base_shap.add_single_point(add_point_x, add_point_y, params={'method': 'avg'})

    * `leave one out` method:

    >>> sv = base_shap.add_single_point(add_point_x, add_point_y, params={'method': 'loo'})

* PivotShap

    >>> pivot_shap = dynashap.PivotShap(x_train, y_train, x_test, y_test, model, init_sv)

    For PivotShap, we need to prepare the Left part Shapley value before add point(s). And ``m`` is the number of the permutations.

    >>> left_sv = pivot_shap.prepare(m)

    This algorithm allows you that add points one by one. And it requires that you need to set the flag `flag_lsv` which means updating the left part Shapley value to True. And `m` is the number of the permutations.

    >>> pivot_sv = pivot_shap.add_single_point(add_point_x, add_point_y, m)

    >>> pivot_shap.add_single_point(add_points_x[0], add_points_y[0], m, {'flag_lsv': True}) # set flag to Ture
    >>> pivot_sv = pivot_shap.add_single_point(add_points_x[1], add_points_y[1], m, {'flag_lsv': True})

* DeltaShap

    >>> delta_shap = dynashap.DeltaShap(x_train, y_train, x_test, y_test, model, init_sv)

    This algorithm allows you that add points one by one as well.

    >>> delta_sv = delta_shap.add_single_point(add_point_x, add_point_y, m)

    >>> delta_shap.add_single_point(add_points_x[0], add_points_y[0], m, {'flag_update': True})
    >>> delta_sv = pivot_shap.add_single_point(add_points_x[1], add_points_y[1], m, {'flag_update': True})

* HeurShap

    Including KNN and KNN+ versions.

    * KNN

    >>> knn_shap = dynashap.HeurShap(x_train, y_train, x_test, y_test, model, init_sv)
    >>> knn_shap.prepare(params={'n_neighbors': 5})
    >>> knn_sv = knn_shap.add_single_point(add_point_x, add_point_y)

    * KNN+

    >>> knn_plus_shap = dynashap.HeurShap(x_train, y_train, x_test, y_test, model, init_sv, params={'method': 'knn+'})
    >>> knn_plus_shap.prepare({'exact': False, 'train': True},
    >>>                       {'n_neighbors': 5, 'simi_type': 'ed', 'f_shap': 'n*n', 'rela': ['poly', 2], 'train_idxs': [...]})
    >>> knn_plus_sv = knn_plus_shap.add_single_point(add_point_x, add_point_y)

Add Multiple Points
-------------------

This method is support by these classes:

* BaseShap

    It is almost the same as what in the `Add Single Point` part.

    >>> sv = base_shap.add_multi_points(add_points_x, add_points_y, params={'method': 'avg'})

    >>> sv = base_shap.add_multi_points(add_points_x, add_points_y, params={'method': 'loo'})

* HeurShap

    Also execute the same preparation shown before.

    >>> knn_sv = knn_shap.add_multi_points(add_points_x, add_points_y)

    >>> knn_plus_sv = knn_plus_shap.add_multi_points(add_points_x, add_points_y)

Delete A Single Point
---------------------

This method is support by these classes:

* YnShap

    >>> yn_shap = dynashap.YnShap(x_train, y_train, x_test, y_test, model, init_sv)

    ``del_num`` is the number of points which need to be deleted. The value of key 'exact' decides that use which approach to calculating Shapley value in the preparation stage.

    >>> yn_shap.prepare(del_num, {'exact': False})
    >>> yn_sv = yn_shap.del_multi_points(del_points_idx)

* DeltaShap

    >>> delta_shap = dynashap.DeltaShap(x_train, y_train, x_test, y_test, model, init_sv)

    This algorithm allows you that delete points one by one as well.

    >>> delta_sv = delta_shap.del_single_point(del_point_idx, m)

    >>> delta_shap.del_single_point(del_points_idx[0], m, {'flag_update': True})
    >>> delta_sv = pivot_shap.del_single_point(del_points_idx[1], m, {'flag_update': True})

* HeurShap

    Preparation can refer to the part `Add Single Point`.

    >>> knn_sv = knn_shap.del_single_point(add_point_x, add_point_y)

    >>> knn_plus_sv = knn_plus_shap.del_single_point(add_point_x, add_point_y)


Delete Multiple Points
----------------------

* YnShap

    >>> yn_shap = dynashap.YnShap(x_train, y_train, x_test, y_test, model, init_sv)

    ``del_num`` is the number of points which need to be deleted. The value of key 'exact' decides that use which approach to calculating Shapley value in the preparation stage.

    >>> yn_shap.prepare(del_num, {'exact': False})
    >>> yn_sv = yn_shap.del_multi_points(del_points_idx)

* HeurShap

    Preparation can refer to the part `Add Single Point`.

    >>> knn_sv = knn_shap.del_multi_point(add_points_x, add_points_y)

    >>> knn_plus_sv = knn_plus_shap.del_multi_point(add_points_x, add_points_y)
