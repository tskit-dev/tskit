---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.12
    jupytext_version: 1.9.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

```{currentmodule} tskit
```

(sec_development)=


# Development

If you would like to add some features to `tskit`, this
documentation should help you get set up and contributing.
Please help us to improve the documentation by either
opening an [issue](http://github.com/tskit-dev/tskit/issues) or
[pull request](http://github.com/tskit-dev/tskit/pulls) if you
see any problems.

The tskit-dev team strives to create a welcoming and open environment for
contributors; please see our
[code of conduct](https://github.com/tskit-dev/.github/blob/main/CODE_OF_CONDUCT.md) for
details. We wish our code and documentation to be
[inclusive](https://chromium.googlesource.com/chromium/src/+/master/styleguide/inclusive_code.md)
and in particular to be gender and racially neutral.

(sec_development_structure)=


## Project structure

Tskit is a multi-language project, which is reflected in the directory
structure:

- The `python` directory contains the Python library and command line interface,
  which is what most contributors are likely to be interested in. Please
  see the {ref}`sec_development_python` section for details. The
  low-level {ref}`sec_development_python_c` is also defined here.

- The `c` directory contains the high-performance C library code. Please
  see the {ref}`sec_development_c` for details on how to contribute.

- The `docs` directory contains the source for this documentation,
  which covers both the Python and C APIs. Please see the {ref}`sec_development_documentation`
  for details.

The remaining files in the root directory of the project are for
controlling {ref}`sec_development_continuous_integration` providers
and other administrative purposes.

Please see the {ref}`sec_development_best_practices` section for
an overview of how to contribute a new feature to `tskit`.

(sec_development_getting_started)=


## Getting started

(sec_development_getting_started_requirements)=


### Requirements

To develop the Python code you will need a working C compiler and a
development installation of Python (>= 3.8). On Debian/Ubuntu we can install these
with:

```bash
$ sudo apt install python3-dev build-essential doxygen
```

Python packages required for development are listed in `python/requirements/development.txt`.
These can be installed using `pip`::

```bash
$ python3 -m pip install -r python/requirements/development.txt
```

You may wish isolate your development environment using a `virtualenv
<https://docs.python-guide.org/dev/virtualenvs/>`_.

A few extra dependencies are required if you wish to work on the
{ref}`C library <sec_development_c_requirements>`.

For OSX and Windows users we recommending using
[conda](https://docs.conda.io/projects/conda/en/latest/)_,
and isolating development in a dedicated environment as follows::

```bash
$ conda env create -f python/requirements/development.yml
$ conda activate tskit-dev
```

On macOS, conda builds are generally done using `clang` packages that are kept up to date:

```bash
$ conda install clang_osx-64  clangxx_osx-64
```

In order to make sure that these compilers work correctly (*e.g.*, so that they can find
other dependencies installed via `conda`), you need to compile `tskit` with this command
on versions of macOS older than "Mojave":

```bash
$ cd python
$ CONDA_BUILD_SYSROOT=/ python3 setup.py build_ext -i
```

On more recent macOS releases, you may omit the `CONDA_BUILD_SYSROOT` prefix.

If you run into issues with the conda compiler, be sure that your command line tools are installed
and up to date (you should also reboot your system after installing CLI tools). Note that you may
also have to install a
[specific version of the Xcode command line tools](https://stackoverflow.com/a/64416852/2752221).

:::{note}
The use of the C toolchain on macOS is a moving target.  The above advice
was updated on 22 June, 2021 and was validated by a few `tskit` contributors.
Caveat emptor, etc..
:::

(sec_development_getting_started_environment)=


### Environment

To get a local git development environment, please follow these steps:

- Make a fork of the tskit repo on [GitHub](http://github.com/tskit-dev/tskit)
- Clone your fork into a local directory:
  ```bash
  $ git clone git@github.com:YOUR_GITHUB_USERNAME/tskit.git
  ```
- Install the {ref}`sec_development_workflow_pre_commit`:
  ```bash
  $ pre-commit install
  ```
  If you later have trouble with these checks, you can skip them with ``git commit --no-verify``.

See the {ref}`sec_development_workflow_git` section for detailed information
on the recommended way to use git and GitHub.

(sec_development_workflow)=


## Workflow

(sec_development_workflow_git)=


### Git workflow

If you would like to make an addition/fix to tskit, then follow the steps below
to get things set up.
If you would just like to review someone else's proposed changes
(either to the code or to the docs), then
skip to {ref}`sec_development_workflow_anothers_commit`.

0.  Open an [issue](http://github.com/tskit-dev/tskit/issues) with your proposed
    functionality/fix. If adding or changing the public API close thought should be given to
    names and signatures of proposed functions. If consensus is reached that your
    proposed addition should be added to the codebase, proceed!

1. Make your own [fork](https://help.github.com/articles/fork-a-repo/)
   of the `tskit` repository on GitHub, and
   [clone](https://help.github.com/articles/cloning-a-repository/)
   a local copy as detailed in {ref}`sec_development_getting_started_environment`.

2. Make sure that your local repository has been configured with an
   [upstream remote](
   https://help.github.com/articles/configuring-a-remote-for-a-fork/):
   ```bash
   $ git remote add upstream https://github.com/tskit-dev/tskit.git
   ```

3. Create a "topic branch" to work on. One reliable way to do it
   is to follow this recipe:
   ```bash
   $ git fetch upstream
   $ git checkout upstream/main
   $ git checkout -b topic_branch_name
   ```

4. Write your code following the outline in {ref}`sec_development_best_practices`.
   As you work on your topic branch you can add commits to it. Once you're
   ready to share this, you can then open a
   [pull request (PR)](https://help.github.com/articles/about-pull-requests/). This can be done at any
   time! You don't have to have code that is completely functional and tested to get
   feedback. Use the drop-down button to create a "draft PR" to indicate that it's not
   done, and explain in the comments what feedback you need and/or what you think needs
   to be done.

5. As you code it is best to
   [rebase](https://stdpopsim.readthedocs.io/en/latest/development.html#rebasing) your
   work onto the `main` branch periodically (e.g. once a week) to keep up with changes.
   If you merge `main` via `git pull upstream main`
   it will create a much more complex rebase when your code is finally ready to be
   incorporated into the main branch, so should be avoided.

6. Once you're done coding add content to the tutorial and other documentation pages if
   appropriate.

7. Update the change logs at `python/CHANGELOG.rst` and `c/CHANGELOG.rst`, taking care
   to document any breaking changes separately in a "breaking changes" section.

8. Push your changes to your topic branch and either open the PR or, if you
   opened a draft PR above change it to a non-draft PR by clicking "Ready to
   Review".

9. The tskit community will review the code, asking you to make changes where appropriate.
   This usually takes at least two rounds of review.

10. Once the review process is complete, squash the commits to the minimal set of changes -
    usually one or two commits. Please follow
    [this guide](https://stdpopsim.readthedocs.io/en/stable/development.html#rebasing) for
    step-by-step instructions on rebasing and squashing commits.

11. Your PR will be merged, time to celebrate! 🎉🍾


(sec_development_workflow_anothers_commit)=


### Checking out someone else's pull request

Sometimes you want to just check out someone else's pull request,
for the purpose of trying it out and giving them feedback.
To do this, you first need your own local version of the git repository,
so you should first do steps 1 and 2 above.
(Strictly speaking, you don't need a fork on github
if you don't plan to edit, but it won't hurt.)
Continuing from there, let's say you want to check out the current
state of the code on [pull request #854](https://github.com/tskit-dev/tskit/pull/854).
(So, below you should replace `854` with the number of the pull request
that you actually want to investigate.)
Then, continuing from above:

3. Fetch the pull request, and store it as a local branch.
   For instance, to name the local branch `my_pr_copy`:
   ```bash
   $ git fetch upstream pull/854/head:my_pr_copy
   ```
   You should probably call the branch something more descriptive,
   though. (Also note that you might need to put `origin` instead
   of `upstream` for the remote repository name: see `git remote -v`
   for a list of possible remotes.)

4. Check out the pull request's local branch:
   ```bash
   $ git checkout my_pr_copy
   ```

Now, your repository will be in exactly the same state as
that of the person who's submitted the pull request.
Great! Now you can test things out.

To view the documentation,
`cd docs && make`, which should build the documentation,
and then navigate your web browser to the `docs/_build/html/`
subdirectory.

To test out changes to the *code*, you can change to the `python/` subdirectory,
and run `make` to compile the C code.
If you then execute `python` from this subdirectory (and only this one!),
it will use the modified version of the package.
(For instance, you might want to
open an interactive `python` shell from the `python/` subdirectory,
or running `python3 -m pytest` from this subdirectory.)

After you're done, you should do:

```bash
$ git checkout main
```

to get your repository back to the "main" branch of development.
If the pull request is changed and you want to do the same thing again,
then first *delete* your local copy (by doing `git branch -d my_pr_copy`)
and repeat the steps again.


(sec_development_workflow_pre_commit)=


### Pre-commit checks

On each commit a [pre-commit hook](https://pre-commit.com/)  will run
that checks for violations of code style
(see the {ref}`sec_development_python_style` section for details)
and other common problems.
Where possible, these hooks will try to fix any problems that they find (including reformatting
your code to conform to the required style). In this case, the commit
will *not complete* and report that "files were modified by this hook".
To include the changes that the hooks made, `git add` any
files that were modified and run `git commit` (or, use `git commit -a`
to commit all changed files.)

If you would like to run the checks without committing, use `pre-commit run`
(but, note that this will *only* check changes that have been *staged*;
do `pre-commit run --all` to check unstaged changes as well).
To bypass the checks (to save or get feedback on work-in-progress) use
`git commit --no-verify`

(sec_development_documentation)=


## Documentation

The documentation for tskit is written using
[Sphinx](http://www.sphinx-doc.org/en/stable/)
and contained in the `docs` directory. The files in this directory are
markdown files that serve as an input to [jupyterbook](https://jupyterbook.org/),
which allows jupyter notebook code, primarily in Python, to be automatically
executed and the output inserted before deployment. The docs are then
deployed automatically to the [tskit.dev website](https://tskit.dev/).
API documentation for both Python and C are generated automatically from
source: documentation embedded in the source code makes use of sphinx and
the [reStructuredText](http://docutils.sourceforge.net/rst.html) format to
alloow formating and cross referencing.
For the C code, a combination of [Doxygen](http://www.doxygen.nl/)
and [breathe](https://breathe.readthedocs.io/en/latest/) is used to
generate API documentation.

Please help us to improve the documentation! You can check on the list of
[documentation issues](https://github.com/tskit-dev/tskit/issues?q=is%3Aissue+is%3Aopen+label%3Adocumentation)
on GitHub, and help us fix any, or add issues for anything that's wrong or missing.


### Small edits

If you see a typo or some other small problem that you'd like to fix,
this is most easily done through the GitHub UI.

If the typo is in a large section of text (like this page), go to the
top of the page and click on the "Edit on GitHub" link at the top
right. This will bring you to the page on GitHub for the RST
source file in question. Then, click on the pencil icon on the
right hand side. This will open a web editor allowing you to
quickly fix the typo and submit a pull request with the changes.
Fix the typo, add a commit message like "Fixed typo" and click
on the green "Propose file change" button. Then follow the dialogues
until you've created a new pull request with your changes,
so that we can incorporate them.

If the change you'd like to make is in the API documentation
for a particular function, then you'll need to find where this
function is defined first. The simplest way to do this is
to click the green "[source]" link next to the function. This
will show you a HTML rendered version of the function, and the
rest of the file that it is in. You can then navigate to this
file on GitHub, and edit it using the same approach as above.


### Significant edits

When making changes more substantial than typo fixes it's best
to check out a local copy.
Follow the steps in the {ref}`sec_development_workflow_git` to
get a fork of tskit, a local clone and newly checked out
feature branch. Then follow the steps in the
{ref}`sec_development_getting_started` section to get a
working development environment.

Once you are ready to make edits to the documentation,
`cd` into the `docs` directory and run `make`.
This should build the HTML
documentation in `docs/_build/html/`, which you can then view in
your browser. As you make changes, run `make` regularly and
view the final result to see if it matches your expectations.

Once you are happy with the changes, commit your updates and
open a pull request on GitHub.


### Tips and resources

- The reStructuredText
  [primer](https://www.sphinx-doc.org/en/1.8/usage/restructuredtext/basics.html)
  is a useful general resource on rst.

- See also the sphinx and rst [cheatsheet
 ](https://docs.typo3.org/m/typo3/docs-how-to-document/master/en-us/WritingReST/CheatSheet.html)

- The Sphinx Python and C
  [domains](https://www.sphinx-doc.org/en/master/usage/restructuredtext/domains.html)
  have extensive options for marking up code.

- Make extensive use of
  [cross referencing](https://docs.typo3.org/m/typo3/docs-how-to-document/master/en-us/WritingReST/Hyperlinks.html).
  When linking to sections in the documentation, use the
  `` :ref:`sec_some_section_label` `` form rather than matching on the section
  title (which is brittle). Use `` :meth:`.Tree.some_method` ``,
  `` :func:`some_function` `` etc to refer to parts of the API.

(sec_development_python)=


## Python library

The Python library is defined in the `python` directory. We assume throughout
this section that you have `cd`'d into this directory.
We also assume that the `tskit` package is built and
run locally *within* this directory. That is, `tskit` is *not* installed
into the Python installation using `pip install -e` or setuptools
[development mode](http://setuptools.readthedocs.io/en/latest/setuptools.html#id23).
Please see the {ref}`sec_development_python_troubleshooting` section for help
if you encounter problems with compiling or running the tests.


### Getting started

After you have installed the basic {ref}`sec_development_getting_started_requirements`
and created a {ref}`development environment <sec_development_getting_started_environment>`,
you will need to compile the low-level {ref}`sec_development_python_c` module.
This is most easily done using `make`:

```bash
$ make
```

If this has completed successfully you should see a file `_tskit.cpython-XXXXXX.so`
in the current directory (the suffix depends on your platform and Python version;
with Python 3.11 on Linux it's `_tskit.cpython-311-x86_64-linux-gnu.so`).

To make sure that your development environment is working, run some
{ref}`tests <sec_development_python_tests>`.


### Layout

Code for the `tskit` module is in the `tskit` directory. The code is split
into a number of modules that are roughly split by function; for example,
code for visualisation is kept in the `tskit/drawing.py`.

Test code is contained in the `tests` directory. Tests are also roughly split
by function, so that tests for the `drawing` module are in the
`tests/test_drawing.py` file. This is not a one-to-one mapping, though.

The `requirements` directory contains descriptions of the requirements
needed for development and on various
{ref}`sec_development_continuous_integration` providers.

(sec_development_python_style)=


### Code style

Python code in tskit is formatted using [Black](https://github.com/psf/black).
Any code submitted as a pull request will be checked to
see if it conforms to this format as part of the
{ref}`sec_development_continuous_integration`. Black is very strict, which seems
unhelpful and nitpicky at first but is actually very useful. This is because it
can also automatically *format* code for you, removing tedious and subjective
choices (and even more tedious and subjective discussions!)
about exactly how a particular code block should be aligned and indented.

In addition to Black autoformatting, code is checked for common problems using
[flake8](https://flake8.pycqa.org/en/latest/)

Black autoformatting and flake8 checks are performed as part of the
{ref}`pre-commit checks <sec_development_workflow_pre_commit>`, which
ensures that your code is always formatted correctly.

Vim users may find the
[black](https://github.com/psf/black),
and
[vim-flake8](https://github.com/nvie/vim-flake8)
plugins useful for automatically formatting code and lint checking
within vim.
There is good support for Black in a number of
[other editors](https://black.readthedocs.io/en/stable/editor_integration.html#editor-integration).


(sec_development_python_tests)=


### Tests

The tests are defined in the `tests` directory, and run using
[pytest](https://docs.pytest.org/en/stable/) from the `python` directory.
If you want to run the tests in a particular module (say, `test_tables.py`), use:

```bash
$ python3 -m pytest tests/test_tables.py
```

To run all the tests in a particular class in this module (say, `TestNodeTable`)
use:

```bash
$ python3 -m pytest tests/test_tables.py::TestNodeTable
```

To run a specific test case in this class (say, `test_copy`) use:

```bash
$ python3 -m pytest tests/test_tables.py::TestNodeTable::test_copy
```

You can also run tests with a keyword expression search. For example this will
run all tests that have `TestNodeTable` but not `copy` in their name:

```bash
$ python3 -m pytest -k "TestNodeTable and not copy"
```

When developing your own tests, it is much quicker to run the specific tests
that you are developing rather than rerunning large sections of the test
suite each time.

To run all of the tests, we can use:

```bash
$ python3 -m pytest
```

By default the tests are run on 4 cores, if you have more you can specify:

```bash
$ python3 -m pytest -n8
```

A few of the tests take most of the time, we can skip the slow tests to get the test run
under 20 seconds on an modern workstation:

```bash
$ python3 -m pytest --skip-slow
```

If you have a lot of failing tests it can be useful to have a shorter summary
of the failing lines:

```bash
$ python3 -m pytest --tb=line
```

If you need to see the output of tests (e.g. `print` statements) then you need to use
these flags to run a single thread and capture output:

```bash
$ python3 -m pytest -n0 -vs
```

All new code must have high test coverage, which will be checked as part of the
{ref}`sec_development_continuous_integration`
tests by [CodeCov](https://codecov.io/gh/tskit-dev/tskit/).
All tests must pass for a PR to be accepted.


### Packaging

The `tskit` Python module follows the current
[best-practices](http://packaging.python.org) advocated by the
[Python Packaging Authority](http://pypa.io/en/latest/). The primary means of
distribution is though [PyPI](http://pypi.python.org/pypi/tskit), which provides the
canonical source for each release.

A package for [conda](http://conda.io/docs/) is also available on
[conda-forge](https://github.com/conda-forge/tskit-feedstock).


### Interfacing with low-level module

Much of the high-level Python code only exists to provide a simpler interface to
the low-level {ref}`_tskit <sec_development_python_c>` module.
As such, many objects (e.g. {class}`.Tree`)
are really just a shallow layer on top of the corresponding low-level object.
The usual convention here is to keep a reference to the low-level object via
a private instance variable such as `self._ll_tree`.


### Command line interface

The command line interface for `tskit` is defined in the `tskit/cli.py` file.
The CLI has a single entry point (e.g. `tskit_main`) which is invoked to run the
program. These entry points are registered with `setuptools` using the
`console_scripts` argument in `setup.py`, which allows them to be deployed as
first-class executable programs in a cross-platform manner.

The CLI can also be run using `python3 -m tskit`. This is the recommended
approach for running the CLI during development.

(sec_development_installing)=

### Installing development versions

We **strongly** recommend that you do not install development versions of
`tskit` and instead use versions released to PyPI and conda-forge.
However, if you really need to be on the bleeding edge, you can use
the following command to install:

```bash
$ python3 -m pip install git+https://github.com/tskit-dev/tskit.git#subdirectory=python
```

(Because the Python package is not defined in the project root directory, using pip to
install directly from  GitHub requires you to specify `subdirectory=python`.)


(sec_development_python_troubleshooting)=

### Troubleshooting

- If `make` is giving you strange errors, or if tests are failing for
  strange reasons, try running `make clean` in the project root
  and then rebuilding.
- Beware of multiple versions of the python library installed by different
  programs (e.g., pip versus installing locally from source)! In python,
  `tskit.__file__` will tell you the location of the package that is being
  used.
- Installation of development version is not supported in Windows. Windows
  users should try using a Linux envronment by using
  [WSL](https://learn.microsoft.com/windows/wsl/), for example.


(sec_development_c)=

## C Library

The Python module uses the high-performance tskit {ref}`sec_c_api`
behind the scenes. All C code and associated development infrastructure
is held in the `c` directory.


(sec_development_c_requirements)=

### Requirements

We use the
[meson](https://mesonbuild.com) build system in conjunction with
[ninja-build](https://ninja-build.org) to compile the C code.
Unit tests use the [CUnit](http://cunit.sourceforge.net) library
and we use [clang-format](https://clang.llvm.org/docs/ClangFormat.html)
to automatically format code.
On Debian/Ubuntu, these can be installed using

```bash
$ sudo apt install libcunit1-dev ninja-build meson clang-format-6.0
```

**Notes:**

1. A more recent version of meson can alternatively be installed using `pip`, if you wish.
2. Recent versions of Debian do not have clang-format-6.0 available;
    if so, you can install it instead with `pip` by running
    `pip3 install clang-format==6.0.1 && ln -s clang-format $(which clang-format)-6.0`.

Conda users can install the basic requirements from `python/requirements/development.txt`.

Unfortunately clang-format is not available on conda, but it is not essential.


(sec_development_c_code_style)=

### Code style

C code is formatted using
[clang-format](https://clang.llvm.org/docs/ClangFormat.html)
with a custom configuration and version 6.0. This is checked as part of the pre-commit
checks. To manually format run:

```bash
$ clang-format-6.0 -i c/tskit/* c/tests/*.c c/tests/*.h
```

Vim users may find the
[vim-clang-format](https://github.com/rhysd/vim-clang-format)
plugin useful for automatically formatting code.


### Building

We use [meson](https://mesonbuild.com) and [ninja-build](https://ninja-build.org) to
compile the C code. Meson keeps all compiled binaries in a build directory (this has many advantages
such as allowing multiple builds with different options to coexist). The build configuration
is defined in `meson.build`. To set up the initial build
directory, run

```bash
$ cd c
$ meson build
```

To compile the code run

```bash
$ ninja -C build
```

All the tests and other artefacts are in the build directory. Individual test
suites can be run, via (e.g.) `./build/test_trees`. To run all of the tests,
run

```bash
$ ninja -C build test
```

For vim users, the [mesonic](https://www.vim.org/scripts/script.php?script_id=5378) plugin
simplifies this process and allows code to be compiled seamlessly within the
editor.


### Unit Tests

The C-library has an extensive suite of unit tests written using
[CUnit](http://cunit.sourceforge.net). These tests aim to establish that the
low-level APIs work correctly over a variety of inputs, and particularly, that
the tests don't result in leaked memory or illegal memory accesses. All tests
are run under valgrind to make sure of this as part of the
{ref}`sec_development_continuous_integration`.

Tests are defined in the `tests/*.c` files. These are roughly split by
the source files, so that the tests for functionality in the `tskit/tables.c` file
will be tested in `tests/test_tables.c`.
To run all the tests
in the `test_tables` suite, run (e.g.) `./build/test_tables`.
To just run a specific test on its own, provide
this test name as a command line argument, e.g.:

```bash
$ ./build/test_tables test_node_table
```

While 100% test coverage is not feasible for C code, we aim to cover all code
that can be reached. (Some classes of error such as malloc failures
and IO errors are difficult to simulate in C.) Code coverage statistics are
automatically tracked using [CodeCov](https://codecov.io/gh/tskit-dev/tskit/).


### Coding conventions

The code is written using the [C99](https://en.wikipedia.org/wiki/C99) standard. All
variable declarations should be done at the start of a function, and functions
kept short and simple where at all possible.

No global or module level variables are used for production code.

Function parameters should be marked as ``const`` where possible.
Parameters that are used as return variables should come last.
The common ``options`` parameter should be the last non-output
parameter.

Please see the {ref}`sec_c_api_overview_structure` section for more information
about how the API is structured.

### Error handling

A critical element of producing reliable C programs is consistent error handling
and checking of return values. All return values **must** be checked! In tskit,
all functions (except the most trivial accessors) return an integer to indicate
success or failure. Any negative value is an error, and must be handled accordingly.
The following pattern is canonical:

```C
   ret = tsk_tree_do_something(self, argument);
    if (ret != 0) {
        goto out;
    }
    // rest of function
out:
    return ret;
```

Here we test the return value of `tsk_tree_do_something` and if it is non-zero,
abort the function and return this same value from the current function. This
is a bit like throwing an exception in higher-level languages, but discipline
is required to ensure that the error codes are propagated back to the original
caller correctly.

Particular care must be taken in functions that allocate memory, because
we must ensure that this memory is freed in all possible success and
failure scenarios. The following pattern is used throughout for this purpose:

```C
    double *x = NULL;

    x = malloc(n * sizeof(double));
    if (x == NULL) {
        ret = TSK_ERR_NO_MEMORY;
        goto out;
    }
    // rest of function
out:
    tsk_safe_free(x);
    return ret;
```

It is vital here that `x` is initialised to `NULL` so that we are guaranteed
correct behaviour in all cases. For this reason, the convention is to declare all
pointer variables on a single line and to initialise them to `NULL` as part
of the declaration.

Error codes are defined in `core.h`, and these can be translated into a
message using `tsk_strerror(err)`.


#### Using assertions

There are two different ways to express assertions in tskit code.
The first is using the custom `tsk_bug_assert` macro, which is used to
make inexpensive checks at key points during execution. These assertions
are always run, regardless of the compiler settings, and should not
contribute significantly to the overall runtime.

More expensive assertions, used, for example, to check pre and post conditions
on performance critical loops should be expressed using the standard
`assert` macro from `assert.h`. These assertions will be checked
during the execution of C unit tests, but will not be enabled when
compiled into the Python C module.


### Type conventions

- `tsk_id_t` is an ID for any entity in a table.
- `tsk_size_t` refers to any size or count values in tskit.
- `size_t` is a standard C type and refers to the size of a memory block.
  This should only be used when computing memory block sizes for functions
  like `malloc` or passing the size of a memory buffer as a parameter.
- Error indicators (the return type of most functions) are `int`.
- `uint32_t` etc should be avoided (any that exist are a leftover from older
  code that didn't use `tsk_size_t` etc.)
- `int64_t` and `uint64_t` are sometimes useful when working with
  bitstrings (e.g. to implement a set).

(sec_development_python_c)=


## Python C Interface


### Overview

The Python C interface is defined in the `python` directory
and written using the [Python C API](https://docs.python.org/3.6/c-api/).
The source code for this interface is in the `_tskitmodule.c` file.
When compiled, this produces the `_tskit` module,
which is imported by the high-level Python code. The low-level Python module is
not intended to be used directly by users and may change arbitrarily over time.

The usual pattern in the low-level Python API is to define a Python class
which corresponds to a given "class" in the C API. For example, we define
a `TreeSequence` class, which is essentially a thin wrapper around the
`tsk_tree_t` type from the C library.

The `_tskitmodule.c` file follows the standard conventions given in the
[Python documentation](https://docs.python.org/3.6/extending/index.html).


### Compiling and debugging

The `setup.py` file describes the requirements for the low-level `_tskit`
module and how it is built from source. The simplest way to compile the
low-level module is to run `make` in the `python` directory:

```bash
$ make
```

If `make` is not available, you can run the same command manually:

```bash
$ python3 setup.py build_ext --inplace
```

It is sometimes useful to specify compiler flags when building the low
level module. For example, to make a debug build you can use:

```bash
$ CFLAGS='-Wall -O0 -g' make
```

If you need to track down a segfault etc, running some code through gdb can
be very useful. For example, to run a particular test case, we can do:


```bash
$ gdb python3
(gdb) run -m pytest tests/test_lowlevel.py


(gdb) run  -m pytest -vs tests/test_tables.py::TestNodeTable::test_copy
Starting program: /usr/bin/python3 run  -m pytest tests/test_tables.py::TestNodeTable::test_copy
[Thread debugging using libthread_db enabled]
Using host libthread_db library "/lib/x86_64-linux-gnu/libthread_db.so.1".
[New Thread 0x7ffff1e48700 (LWP 1503)]
[New Thread 0x7fffef647700 (LWP 1504)]
[New Thread 0x7fffeee46700 (LWP 1505)]
[Thread 0x7fffeee46700 (LWP 1505) exited]
[Thread 0x7fffef647700 (LWP 1504) exited]
[Thread 0x7ffff1e48700 (LWP 1503) exited]
collected 1 item

tests/test_tables.py::TestNodeTable::test_copy PASSED

[Inferior 1 (process 1499) exited normally]
(gdb)
```

Tracing problems in C code is many times more difficult when the Python C API
is involved because of the complexity of Python's memory management. It is
nearly always best to start by making sure that the tskit C API part of your
addition is thoroughly tested with valgrind before resorting to the debugger.


### Testing for memory leaks

The Python C API can be subtle, and it is easy to get the reference counting wrong.
The `stress_lowlevel.py` script makes it easier to track down memory leaks
when they do occur. The script runs the unit tests in a loop, and outputs
memory usage statistics.


(sec_development_continuous_integration)=


## Continuous Integration tests

A number of different continuous integration providers are used, which run different
combinations of tests on different platforms, as well as running various
checks for code quality.

- A [Github action](https://help.github.com/en/actions) runs some code style and
  quality checks along with running the Python test suite on Linux, OSX and Windows. It
  uses conda for those dependencies which are tricky to compile on all systems. An
  additional action builds the docs and posts a link to preview them.

- [CircleCI](https://circleci.com/) runs all Python tests using the apt-get
  infrastructure for system requirements. We also runs C tests, compiled
  using gcc and clang, and check for memory leaks using valgrind.

- [CodeCov](https://codecov.io/gh)_ tracks test coverage in Python and C.

- [PyUp](https://pyup.io/) Runs monthly checks on the Python dependencies listed in the
  requirements files, which are pinned to ensure CI reproducibility. PyUp opens one PR
  a month with updated pins.


(sec_development_best_practices)=


## Best Practices for Development

The following is a rough guide of best practices for contributing a function to the
tskit codebase.

Note that this guide covers the most complex case of adding a new function to both
the C and Python APIs.

1.  Write your function in Python: in `python/tests/` find the test module that
    pertains to the functionality you wish to add. For instance, the kc_distance
    metric was added to
    [test_topology.py](https://github.com/tskit-dev/tskit/blob/main/python/tests/test_topology.py).
    Add a python version of your function here.
2.  Create a new class in this module to write unit tests for your function: in addition
    to making sure that your function is correct, make sure it fails on inappropriate inputs.
    This can often require judgement. For instance, {meth}`Tree.kc_distance` fails on a tree
    with multiple roots, but allows users to input parameter values that are nonsensical,
    as long as they don't break functionality. See the
    [TestKCMetric](https://github.com/tskit-dev/tskit/blob/4e707ea04adca256036669cd852656a08ec45590/python/tests/test_topology.py#L293) for example.
3.  Write your function in C: check out the {ref}`sec_c_api` for guidance. There
    are also many examples in the
    [c directory](https://github.com/tskit-dev/tskit/tree/main/c/tskit).
    Your function will probably go in
    [trees.c](https://github.com/tskit-dev/tskit/blob/main/c/tskit/trees.c).
4.  Write a few tests for your function in C: again, write your tests in
    [tskit/c/tests/test_tree.c](https://github.com/tskit-dev/tskit/blob/main/c/tests/test_trees.c).
    The key here is code coverage, you don't need to worry as much about covering every
    corner case, as we will proceed to link this function to the Python tests you
    wrote earlier.
5.  Create a low-level definition of your function using Python's C API: this will
    go in [_tskitmodule.c](https://github.com/tskit-dev/tskit/blob/main/python/_tskitmodule.c).
6.  Test your low-level implementation in [tskit/python/tests/test_lowlevel.py
   ](https://github.com/tskit-dev/tskit/blob/main/python/tests/test_lowlevel.py):
    again, these tests don't need to be as comprehensive as your first python tests,
    instead, they should focus on the interface, e.g., does the function behave
    correctly on malformed inputs?
7.  Link your C function to the Python API: write a function in tskit's Python API,
    for example the kc_distance function lives in
    [tskit/python/tskit/trees.py](https://github.com/tskit-dev/tskit/blob/main/python/tskit/trees.py).
8.  Modify your Python tests to test the new C-linked function: if you followed
    the example of other tests, you might need to only add a single line of code
    here. In this case, the tests are well factored so that we can easily compare
    the results from both the Python and C versions.
9.  Write a docstring for your function in the Python API: for instance, the kc_distance
    docstring is in
    [tskit/python/tskit/trees.py](https://github.com/tskit-dev/tskit/blob/main/python/tskit/trees.py).
    Ensure that your docstring renders correctly by building the documentation
    (see {ref}`sec_development_documentation`).


## Troubleshooting

### pre-commit is blocking me!

You might be having a hard time committing because of the "pre-commit" checks
(described above). First, consider: the pre-commit hooks are supposed to make your life *easier*,
not add a layer of frustration to contributing.
So, you should feel free to just ask git to skip the pre-commit!
There's no shame in a broken build - you can get it fixed up (and we'll help)
before it's merged into the rest of the project.
To skip, just append `--no-verify` to the `git commit` command.
Below are some more specific situations.


### pre-commit complains about files I didn't edit

For instance, suppose you have *not* edited `util.py` and yet:

```bash
> git commit -a -m 'dev docs'
python/tskit/util.py:117:26: E203 whitespace before ':'
python/tskit/util.py:135:31: E203 whitespace before ':'
python/tskit/util.py:195:23: E203 whitespace before ':'
python/tskit/util.py:213:36: E203 whitespace before ':'
... lots more, gah, what is this ...
```

First, check (with `git status`) that you didn't actually edit `util.py`.
Then, you should **not** try to fix these errors; this is **not your problem**.
You might first try restarting your pre-commit, by running

```bash
pre-commit clean
pre-commit gc
```

You might also check you don't have other pre-commit hook files in `.git/hooks`.
If this doesn't fix the problem,
then you should just *skip* the pre-commit (but alert us to the problem),
by appending `--no-verify`:

```bash
> git commit -a -m 'dev docs' --no-verify
[main 46f3f2e] dev docs
 1 file changed, 43 insertions(+)
```

Now you can go ahead and push your changes!


### pre-commit won't run

For instance:

```bash
> git commit -a -m 'fixed all the things'
/usr/bin/env: ‘python3.8’: No such file or directory
```

What the heck? Why is this even looking for python3.8?
This is because of the "pre-commit hook", mentioned above.
As above, you can proceed by just appending `--no-verify`:

```bash
> git commit -a -m 'fixed all the things' --no-verify
[main 99a01da] fixed all the things
 1 file changed, 10 insertions(+)
```

We'll help you sort it out in the PR.
But, you should fix the problem at some point. In this case,
uninstalling and reinstalling the pre-commit hooks fixed the problem:

```bash
> pre-commit uninstall
pre-commit uninstalled
Restored previous hooks to .git/hooks/pre-commit
> pre-commit install -f
pre-commit installed at .git/hooks/pre-commit
> # do some more edits
> git commit -a -m 'wrote the docs'
[main 79b81ff] fixed all the things
 1 file changed, 42 insertions(+)
```


## Benchmarking

Tskit has a simple benchmarking tool to help keep track of performance.

### Running benchmarks

The benchmark suite can be run with:

```bash
> cd python/benchmark
> python run.py
```

A subset of benchmarks can be run by specifying a string. For example, the following command runs all the benchmarks whose names contain "genotype", e.g. "genotype_matrix".

```bash
> python run.py -k genotype
```

If desired, the results of the benchmarks can be printed to STDOUT.

```bash
> python run.py -k genotype -p
```

Results are written to `bench-results.json` in the same folder. Note that if any version of `tskit`
is installed then that will be used for the benchmarking. To use the local development version of
tskit ensure you have `pip uninstall tskit` before running the benchmarking. The version used is
shown in the header of the report.

### Adding a new benchmark

The benchmarks are specified by the `config.yaml` file in `python/benchmark`. To add a new benchmark 
add an entry to the `benchmarks` dictionary. For example:

```yaml
  - code: do_my_thing({option_name})
    setup: |
      import a_module
    name: my_benchmark #optional, the code is used by default
    parameters:
      option_name:
        - "reticulate_splines"
        - "foobar"
```

Strings are interpreted as Python f-strings, so you can use the `parameters` dictionary to provide
values that will be interpolated into both the `setup` and `code` strings.

The suite can be run for all released versions with the `run-for-all-releases.py` script. 

## Releasing a new version

Tskit maintains separate versioning for the C API and Python package, each has its own
release process.


### C API

To release the C API, the ``TSK_VERSION_*`` macros should be updated,
along with ``VERSION.txt`` and the changelog  updated with the release
date and version. The changelog should also be checked for
completeness. Comparing  ``git log --follow --oneline -- c`` with
`git log --follow --oneline -- c/CHANGELOG.rst` may help here.
After the commit including these changes has been merged, tag a
release on GitHub using the pattern `C_MAJOR.MINOR.PATCH`, with:

```bash
git tag -a C_MAJOR.MINOR.PATCH -m "C API version C_MAJOR.MINOR.PATCH"
git push upstream --tags
```

After a couple of minutes a github action will make a draft release with the changelog
at the [releases page](https://github.com/tskit-dev/tskit/releases). Check it looks
right and publish the release (Click on the little pencil).
After release, start a section in the changelog for new developments and close the
GitHub issue milestone of the release.


### Python

It is worth running the benchmarks as above before release to check for any unexpected
major regressions. To make a release first prepare a pull request that sets the correct
version number in `tskit/_version.py`  following PEP440 format. For a normal release
this should be MAJOR.MINOR.PATCH, for a beta release use MAJOR.MINOR.PATCHbX
e.g. 1.0.0b1. Update the Python CHANGELOG.rst, ensuring that all significant
changes since the last release have been listed. Comparing
`git log --follow --oneline -- python`
with `git log --follow --oneline -- python/CHANGELOG.rst` may help here.
Once this PR is merged, push a tag to github:

```bash
git tag -a MAJOR.MINOR.PATCH -m "Python version MAJOR.MINOR.PATCH"
git push upstream --tags
```

This will trigger a build of the distribution artifacts for Python
on [Github Actions](https://github.com/tskit-dev/tskit/actions). and deploy
them to the [test PyPI](https://test.pypi.org/project/tskit/). Check
the release looks good there, then publish the draft release on the
[releases page](https://github.com/tskit-dev/tskit/releases) (Click on the little pencil).
Publishing this release will cause the github
action to deploy to the [production PyPI](https://pypi.org/project/tskit/).
After release, start a section in the changelog for new developments, close the
GitHub issue milestone of the release and update ROADMAP.md.
For a major release the website (github repo tskit-dev/tskit-site) should then
be updated with a notebook of new features. The benchmarks should be run as above
and the `bench-results.html` updated on the website.
