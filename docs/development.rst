.. _sec_development:

===========
Development
===========

If you want to try out the development version of ``tskit``, or add features or
documentation improvements, please read the following. If you think there is anything
missing, please open an `issue <http://github.com/tskit-dev/tskit/issues>`_ or
`pull request <http://github.com/tskit-dev/tskit/pulls>`_ on GitHub!

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
