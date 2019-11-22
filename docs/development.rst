.. currentmodule:: tskit
.. _sec_development:

===========
Development
===========

If you want to try out the development version of ``tskit``, or add features or
documentation improvements, please read the following. If you think there is anything
missing, please open an `issue <http://github.com/tskit-dev/tskit/issues>`_ or
`pull request <http://github.com/tskit-dev/tskit/pulls>`_ on GitHub!

.. _sec_development_quickstart:

**********
Quickstart
**********

- Make a fork of the tskit repo on `GitHub <http://github.com/tskit-dev/tskit>`_
- Clone your fork into a local directory, making sure that the **submodules
  are correctly initialised**::

  $ git clone git@github.com:tskit-dev/tskit.git --recurse-submodules

  For an already checked out repo, the submodules can be initialised using::

  $ git submodule update --init --recursive

  (note that on Windows you may have to run this as administrator, in order to create the
  correct symlinks)
- Install the Python development requirements using
  ``pip install -r python/requirements/development.txt``.
- Build the low level module by running ``make`` in the ``python`` directory.
- Run the tests to ensure everything has worked: ``python -m nose -vs``. These should
  all pass.

- Make your changes in a local branch, and open a pull request on GitHub when you
  are ready. Please make sure that (a) the tests pass before you open the PR; and
  (b) your code passes PEP8 checks (see :ref:`sec_development_code_style` below
  for a git commit hook to ensure this happens automatically) before opening the PR.

****************************
Continuous integration tests
****************************

Three different continuous integration providers are used, which run different
combinations of tests on different platforms:

1. `Travis CI <https://travis-ci.org/>`_ runs tests on Linux and OSX using the
   `Conda <https://conda.io/docs/>`__ infrastructure for the system level
   requirements. All supported versions of Python are tested here.

2. `CircleCI <https://circleci.com/>`_ Runs all Python tests using the apt-get
   infrastructure for system requirements. Additionally, the low-level tests
   are run, coverage statistics calculated using `CodeCov <https://codecov.io/gh>`__,
   and the documentation built.

3. `AppVeyor <https://www.appveyor.com/>`_ Runs Python tests on 32 and 64 bit
   Windows using conda.

.. _sec_development_code_style:

**********
Code style
**********

Submitted code should conform to `PEP8 <https://www.python.org/dev/peps/pep-0008/>`_.
An easy way to ensure this is the case is to add a ``pre-commit`` file to the
``.git/hooks/`` directory in your local installation. We recommending adding something
like the following line to your ``pre-commit`` file:

``exec flake8 python/setup.py python/tskit python/tests --max-line-length=89``

(the flake8 python package should have been installed as part of the Python development
requirements above)

.. _sec_development_installing:

*******************************
Installing development versions
*******************************

Because the python package is not defined in the project root directory, using pip to 
install the bleeding-edge package directly from the tskit github repository requires you
to specify ``subdirectory=python``:

``python3 -m pip install git+https://github.com/tskit-dev/tskit.git#subdirectory=python``

******************************
Best Practices for Development
******************************

The following is a rough guide of best practices for contributing a function to the
tskit codebase.
We will use the :meth:`Tree.kc_distance` method as an example. 

0.  Open an `issue <http://github.com/tskit-dev/tskit/issues>`_ with your proposed
    functionality. If consensus is reached that your proposed addition should be
    added to the codebase, proceed! 
1.  Create a new branch on your fork of tskit-dev (see :ref:`sec_development_quickstart`
    above). Then open a `pull request <http://github.com/tskit-dev/tskit/pulls>`_ on GitHub,
    with an initial description of your planned addition.
2.  Write your function in Python: in python/tests find the test module that 
    pertains to the functionality you wish to add. For instance, the kc_distance
    metric was added to 
    `test_topology.py <https://github.com/tskit-dev/tskit/blob/master/python/tests/test_topology.py>`_.
    Add a python version of your function here.
3.  Create a new class in this module to write unit tests for your function: in addition
    to making sure that your function is correct, make sure it fails on inappropriate inputs.
    This can often require judgement. For instance, :meth:`Tree.kc_distance` fails on a tree 
    with multiple roots, but allows users to input parameter values that are nonsensical,
    as long as they don't break functionality. 
4.  Write your function in C: check out the :ref:`sec_c_api` for guidance. There
    are also many examples in the 
    `c directory <https://github.com/tskit-dev/tskit/tree/master/c/tskit>`_.
    Your function will probably go in 
    `trees.c <https://github.com/tskit-dev/tskit/blob/master/c/tskit/trees.c>`_.
5.  Write a few tests for your function in C: again, write your tests in  
    `tskit/c/tests/test_tree.c <https://github.com/tskit-dev/tskit/blob/master/c/tests/test_trees.c>`_.
    The key here is code coverage, you don't need to worry as much about covering every
    corner case, as we will proceed to link this function to the Python tests you
    wrote earlier.
6.  Create a low-level definition of your function using Python's C API: this will
    go in `_tskitmodule.c 
    <https://github.com/tskit-dev/tskit/blob/master/python/_tskitmodule.c>`_.
7.  Test your low-level implementation in `tskit/python/tests/test_lowlevel.py
    <https://github.com/tskit-dev/tskit/blob/master/python/tests/test_lowlevel.py>`_:
    again, these tests don't need to be as comprehensive as your first python tests,
    instead, they should focus on the interface, e.g., does the function behave
    correctly on malformed inputs?
8.  Link your C funtion to the Python API: write a function in tskit's Python API, 
    for example the kc_distance function lives in 
    `tskit/python/tskit/trees.py 
    <https://github.com/tskit-dev/tskit/blob/master/python/tskit/trees.py>`_.
9.  Modify your Python tests to test the new C-linked function: if you followed
    the example of other tests, you might need to only add a single line of code
    here.
10. Write a docstring for your function in the Python API: for instance, the kc_distance
    docstring is in 
    `tskit/python/tskit/trees.py 
    <https://github.com/tskit-dev/tskit/blob/master/python/tskit/trees.py>`_.
    Ensure that your docstring renders correctly by building the documentation
    (see :ref:`sec_development_documentation`).
11. Update your Pull Request (`rebasing <https://stdpopsim.readthedocs.io/en/
    latest/development.html#rebasing>`_ if necessary!) and let the community check
    your work.

.. _sec_development_documentation:

*************
Documentation
*************

TODO
