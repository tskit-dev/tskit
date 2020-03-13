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
- Install the pre-commit checks: ``pre-commit install``
- Make your changes in a local branch. On each commit a `pre-commit hook
  <https://pre-commit.com/>`_  will run
  checks for code style and common problems (see :ref:`sec_development_code_style`).
  Sometimes these will report "files were modified by this hook" ``git add``
  and ``git commit --amend`` will update the commit with the automatically modified
  version.
  The modifications made are for consistency, code readability and designed to
  minimise merge conflicts. They are guaranteed not to modify the functionality of the
  code. To run the checks without committing use ``pre-commit run``. To bypass
  the checks (to save or get feedback on work-in-progress) use ``git commit
  --no-verify``
- If you have modified C code then run this to make it conform to the project style::

  $ sudo apt-get install clang-format
  $ clang-format -i c/tskit/* c/tests/*.c c/tests/*.h

- When ready open a pull request on GitHub. Please make sure that the tests pass before
  you open the PR, unless you want to ask the community for help with a failing test.

****************************
Continuous integration tests
****************************

Three different continuous integration providers are used, which run different
combinations of tests on different platforms:

1. A `Github action <https://help.github.com/en/actions>`_ runs `pre-commit
   <https://pre-commit.com/>`_ to run a variety of code style and quality checks.

2. `Travis CI <https://travis-ci.org/>`_ runs tests on Linux and OSX using the
   `Conda <https://conda.io/docs/>`__ infrastructure for the system level
   requirements. All supported versions of Python are tested here.

3. `CircleCI <https://circleci.com/>`_ Runs all Python tests using the apt-get
   infrastructure for system requirements. Additionally, the low-level tests
   are run, coverage statistics calculated using `CodeCov <https://codecov.io/gh>`__,
   and the documentation built.

4. `AppVeyor <https://www.appveyor.com/>`_ Runs Python tests on 32 and 64 bit
   Windows using conda.

.. _sec_development_code_style:

**********
Code style
**********

Submitted python code should conform to `PEP8 <https://www.python
.org/dev/peps/pep-0008/>`_. `Black <https://github.com/psf/black>`_ and
`flake8 <https://flake8.pycqa.org/en/latest/>`_ are used as part of the pre-commit
hook for python code style and formatting.

C code is formatted using `clang-format` with a custom
configuration: `clang-format -i c/tskit/* c/tests/*.c c/tests/*.h`

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
