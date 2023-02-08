# -*- coding: utf-8 -*-
# Copyright (c) Haocheng Xia.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
dynashap.exceptions
~~~~~~~~~~~~~~~~~~~
This module contains the set of exceptions.
"""


class UnImpException(Exception):
    """Method Unimplemented Exception."""

    def __init__(self, name) -> None:
        self.name = name

    def __str__(self) -> None:
        print('method' + str(self.name) + 'is NOT implemented')


class ParamError(ValueError, TypeError):
    """The parameter is missing or mismatching."""


class FlagError(ValueError, TypeError):
    """The flag is missing or mismatching."""


# Warnings

class StepWarning(Warning):
    """Base warning for invalid step."""
