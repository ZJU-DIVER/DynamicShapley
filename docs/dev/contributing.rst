.. _contributing:

===================
Contributor's Guide
===================

If you're reading this, you're probably interested in contributing to DynaShap.
Thank you very much!

Code Contributions
==================

Steps for Submitting Code
~~~~~~~~~~~~~~~~~~~~~~~~~

When contributing code, you'll want to follow this checklist:

1. Fork the repository on GitHub.
2. Make your change and describe it.
3. Run the entire examples, confirming that all examples pass *including
   the ones you just added*.
4. Send a GitHub Pull Request to the main repository's ``master`` branch.
   GitHub Pull Requests are the expected method of code collaboration on this
   project.

The following sub-sections go into more detail on some of the points above.

Code Review
~~~~~~~~~~~

Contributions will not be merged until they've been code reviewed. You should
implement any code review feedback unless you strongly object to it. In the
event that you object to the code review feedback, you should make your case
clearly and calmly. If, after doing so, the feedback is judged to still apply,
you must either apply the feedback or withdraw your contribution.

Code Style
~~~~~~~~~~

The DynaShap codebase uses the `PEP 8`_ code style.

In addition to the standards outlined in PEP 8, we have a few guidelines:

- Line-length can exceed 79 characters, to 100, when convenient.
- Line-length can exceed 100 characters, when doing otherwise would be *terribly* inconvenient.
- Always use single-quoted strings (e.g. ``'#flatearth'``), unless a single-quote occurs within the string.

Additionally, one of the styles that PEP8 recommends for `line continuations`_
completely lacks all sense of taste, and is not to be permitted within
the Requests codebase::

    # Aligned with opening delimiter.
    foo = long_function_name(var_one, var_two,
                             var_three, var_four)

No. Just don't. Please. This is much better::

    foo = long_function_name(
        var_one,
        var_two,
        var_three,
        var_four,
    )

Docstrings are to follow the following syntaxes::

    def the_earth_is_flat():
        """NASA divided up the seas into thirty-three degrees."""
        pass

::

    def fibonacci_spiral_tool():
        """With my feet upon the ground I lose myself / between the sounds
        and open wide to suck it in. / I feel it move across my skin. / I'm
        reaching up and reaching out. / I'm reaching for the random or
        whatever will bewilder me. / Whatever will bewilder me. / And
        following our will and wind we may just go where no one's been. /
        We'll ride the spiral to the end and may just go where no one's
        been.

        Spiral out. Keep going...
        """
        pass

All functions, methods, and classes are to contain docstrings. Object data
model methods (e.g. ``__repr__``) are typically the exception to this rule.

Thanks for helping to make the world a better place!

.. _PEP 8: https://pep8.org/
.. _line continuations: https://www.python.org/dev/peps/pep-0008/#indentation