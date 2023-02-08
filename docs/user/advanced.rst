.. _advanced:

Advanced Usage
==============

You can refer to the source code and directly operate the variables to
avoid repeating computation.

.. note::
    There is a simple example for KNN+ algorithm (Heuristic Shapley).

    >>> idxs = [...]
    >>> svs = list()
    >>> for idx in idxs:
    >>>     sv = dynashap.mc_shap(np.delete(x_train, idx, axis=0), np.delete(y_train, idx), x_test_added, y_test_added, model, m)
    >>>     svs.append(sv)
    >>> svs = np.asarray(svs)
    >>> knn_plus_shap = dynashap.HeurShap(x_train, y_train, x_test, y_test, model, init_sv, params={'method': 'knn+'})
    >>> knn_plus_shap.prepare({'exact': False, 'train': False},
    >>>                       {'n_neighbors': 5, 'simi_type': 'ed', 'f_shap': 'n*n', 'rela': ['poly', 1], 'train_idxs': [...], 'train_svs': svs})
    >>> knn_plus_sv = knn_plus_shap.add_single_point(add_point_x, add_point_y)