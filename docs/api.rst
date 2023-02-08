=========
Interface
=========

.. module:: dynashap

This part of the documentation covers all the interfaces of DynaShap. For parts where Requests depends on external libraries, we document the most important right here and provide links to the canonical documentation.

Main Interface
==============

All of Requests' functionality can be accessed by these 5 classes of methods. They all return an instance of the :class:`numpy.ndarray <numpy.ndarray>` object which contains the Shapley value excepting `prepare`.

Add Single Point
~~~~~~~~~~~~~~~~

.. autofunction:: dynashap.BaseShap.add_single_point
.. autofunction:: dynashap.PivotShap.add_single_point
.. autofunction:: dynashap.DeltaShap.add_single_point
.. autofunction:: dynashap.HeurShap.add_single_point

Add Multiple Point
~~~~~~~~~~~~~~~~~~

.. autofunction:: dynashap.BaseShap.add_multi_points
.. autofunction:: dynashap.HeurShap.add_multi_points

Delete Single Point
~~~~~~~~~~~~~~~~~~~

.. autofunction:: dynashap.YnShap.del_single_point
.. autofunction:: dynashap.DeltaShap.del_single_point
.. autofunction:: dynashap.HeurShap.del_single_point

Delete Multiple Point
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: dynashap.YnShap.del_multi_points
.. autofunction:: dynashap.HeurShap.del_multi_points

Prepare
~~~~~~~

.. autofunction:: dynashap.YnShap.prepare
.. autofunction:: dynashap.PivotShap.prepare
.. autofunction:: dynashap.HeurShap.prepare

Exceptions
==========

.. autoexception:: dynashap.UnImpException
.. autoexception:: dynashap.ParamError
.. autoexception:: dynashap.FlagError
.. autoexception:: dynashap.StepWarning