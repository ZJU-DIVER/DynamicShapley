# -*- coding: utf-8 -*-
# Copyright (c) Haocheng Xia.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
dynashap.structures
~~~~~~~~~~~~~~~~~~~
Data structures that power DynaShap.
"""


class SimiPreData(object):

    def __init__(self, params) -> None:
        self.train_idxs = params['train_idxs']
        self.svs = params['train_svs']
