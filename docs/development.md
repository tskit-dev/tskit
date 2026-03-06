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

(sec_development_repo_admin)=


## Repo administration

tskit is one of several packages in the tskit-dev ecosystem. Shared conventions
for CI workflows, dependency management, repository layout, and releases are
documented in the
[repo administration guide](https://github.com/tskit-dev/.github/blob/main/repo_administration.md)
in the `tskit-dev/.github` repository. Maintainers should read that document
before making changes to CI configuration, dependency groups, or the release process.


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
and some build utilities. Additionally, the doxygen package
is required for building the C API documentation.
On Debian/Ubuntu we can install these with:

```bash
$ sudo apt install build-essential doxygen
```

All Python development is managed using [uv](https://docs.astral.sh/uv/).
It is not strictly necessary to use uv in order to make small changes, but
the development workflows of all tskit-dev packages are organised around
using uv, and therefore we strongly recommend using it. Uv is straightforward
to install, and not invasive (existing Python installations can be completely
isolated if you don't use features like ``uv tool`` etc which update your
$HOME/.local/bin). Uv manages an isolated local environment per project
and allows us to deterministically pin package versions and easily switch
between Python versions, so that CI environments can be replicated exactly
locally.

The packages needed for development are specified as dependency groups
in ``python/pyproject.toml`` and managed with [uv](https://docs.astral.sh/uv/).
Install all development dependencies using:

```bash
$ uv sync
```

The lock file lives at `python/uv.lock` and must be kept up to date. Run
`uv lock` after any change to the dependencies in `python/pyproject.toml`.

A few extra dependencies are required if you wish to work on the
{ref}`C library <sec_development_c_requirements>`.

(sec_development_getting_started_environment)=


### Environment

To get a local git development environment, please follow these steps:

- Make a fork of the tskit repo on [GitHub](http://github.com/tskit-dev/tskit)
- Clone your fork into a local directory:
  ```bash
  $ git clone git@github.com:YOUR_GITHUB_USERNAME/tskit.git
  ```
- Install the {ref}`sec_development_workflow_prek` pre-commit hook:
  ```bash
  $ uv run prek install
  ```

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
or running `uv run pytest` from this subdirectory.)

After you're done, you should do:

```bash
$ git checkout main
```

to get your repository back to the "main" branch of development.
If the pull request is changed and you want to do the same thing again,
then first *delete* your local copy (by doing `git branch -d my_pr_copy`)
and repeat the steps again.


(sec_development_workflow_prek)=


### Lint checks (prek)

On each commit a [prek](https://prek.j178.dev) hook will run checks for
code style (see the {ref}`sec_development_python_style` section for details)
and other common problems.

To install the hook:

```bash
$ uv run prek install
```

To run checks manually without committing:

```bash
$ uv run prek --all-files
```

If local results differ from CI, run `uv run prek cache clean` to clear the cache.
To bypass the checks temporarily use `git commit --no-verify`.

(sec_development_documentation)=


## Documentation

The documentation for tskit is written using
[Sphinx](http://www.sphinx-doc.org/en/stable/) and contained in the `docs`
directory. Narrative pages are written in
[MyST Markdown](https://jupyterbook.org/content/myst.html) and built with
[JupyterBook](https://jupyterbook.org/), which executes embedded Python code
cells and inserts their output before deployment. API docstrings are written in
[reStructuredText](http://docutils.sourceforge.net/rst.html). For the C code,
a combination of [Doxygen](http://www.doxygen.nl/) and
[breathe](https://breathe.readthedocs.io/en/latest/) generates API documentation.
The docs are deployed automatically to the [tskit.dev website](https://tskit.dev/).

Please help us to improve the documentation! You can check on the list of
[documentation issues](https://github.com/tskit-dev/tskit/issues?q=is%3Aissue+is%3Aopen+label%3Adocumentation)
on GitHub, and help us fix any, or add issues for anything that's wrong or missing.


### Small edits

If you see a typo or some other small problem that you'd like to fix,
this is most easily done through the GitHub UI.

Mouse over the GitHub icon at the top right of the page and
click on the "Suggest edit" button. This will bring you to a web
editor on GitHub for the source file in question, allowing you to
quickly fix the typo and submit a pull request with the changes.
Fix the typo, click the "Commit changes", add a commit message like
"Fixed typo" and click on the green "Propose file change" button.
Then follow the dialogues until you've created a new pull request
with your changes, so that we can incorporate them.

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


(sec_development_documentation_markup)=

### Markup languages

Because of the mixture of API documentation and notebook content, documentation
is written using **two different markup languages**:

- **MyST Markdown** for all narrative pages, thematic sections, and code
  examples. This is a superset of [CommonMark](https://commonmark.org) that
  enables executable Jupyter content and Sphinx cross-referencing.
- **reStructuredText (rST)** for API docstrings embedded in the source code.
  These are processed by Sphinx and appear in the API reference pages.

Some useful links for MyST:

- The [MyST cheat sheet](https://jupyterbook.org/reference/cheatsheet.html)
- The "Write Book Content" section of the [Jupyter Book](https://jupyterbook.org/) docs
- The [MyST Syntax Guide](https://myst-parser.readthedocs.io/en/latest/using/syntax.html)
- The [Sphinx domains reference](https://www.sphinx-doc.org/en/master/usage/restructuredtext/domains.html)
  for marking up Python and C API elements
- The [types of source files](https://jupyterbook.org/file-types/index.html)
  in the Jupyter Book docs (useful for understanding the MyST/rST mix)

Some directives are only available in rST and must be wrapped in an
``eval-rst`` block within a Markdown file:

````md
```{eval-rst}
.. autoclass:: tskit.TreeSequence
```
````

(sec_development_documentation_api)=

### API Reference

API reference documentation comes from
[docstrings](https://www.python.org/dev/peps/pep-0257/) in the source code,
written in rST. Docstrings should be **concise** and **precise**. Examples
should not be embedded directly in docstrings; instead, each significant
parameter should link to the relevant section in the narrative documentation.

(sec_development_documentation_examples)=

### Examples

Narrative sections should provide context and worked examples using inline
Jupyter code cells. These behave exactly like cells in a Jupyter notebook —
the whole page is executed as one notebook during the build.

Code cells are written like this:

````md
```{code-cell}
import tskit
# example code here
```
````

:::{warning}
For a page to be executed as a notebook you **must** have the correct
[YAML frontmatter](https://jupyterbook.org/reference/cheatsheet.html#executable-code)
at the top of the file.
:::

(sec_development_documentation_cross_referencing)=

### Cross referencing

Use the ``{ref}`` role to link to labelled sections within the docs:

````md
See the {ref}`sec_development_documentation_cross_referencing` section for details.
````

Sections should be labelled hierarchically immediately above the heading:

````md
(sec_development_documentation_cross_referencing)=
### Cross referencing
````

The label is used as link text automatically, but can be overridden:

````md
See {ref}`this section <sec_development_documentation_cross_referencing>` for more.
````

To refer to API elements, use the appropriate inline role:

````md
The {class}`.TreeSequence` class, the {meth}`.TreeSequence.trees` method,
and the {func}`.load` function.
````

From an rST docstring, use the colon-prefixed equivalents:

````rst
See :ref:`sec_development_documentation_cross_referencing` for details.
The :meth:`.TreeSequence.trees` method returns an iterator.
````

(sec_development_python)=


## Python library

The Python library is defined in the `python` directory. We assume throughout
this section that you have `cd`'d into this directory.
The low-level C extension is built automatically as part of `uv sync`.
Please see the {ref}`sec_development_python_troubleshooting` section for help
if you encounter problems with compiling or running the tests.


### Getting started

After you have installed the basic {ref}`sec_development_getting_started_requirements`
and created a {ref}`development environment <sec_development_getting_started_environment>`,
run `uv sync` at the repo root. This will install all dependencies and build
the low-level {ref}`sec_development_python_c` module automatically.

To make sure that your development environment is working, run some
{ref}`tests <sec_development_python_tests>`.


### Layout

Code for the `tskit` module is in the `tskit` directory. The code is split
into a number of modules that are roughly split by function; for example,
code for visualisation is kept in the `tskit/drawing.py`.

Test code is contained in the `tests` directory. Tests are also roughly split
by function, so that tests for the `drawing` module are in the
`tests/test_drawing.py` file. This is not a one-to-one mapping, though.

Development dependencies are specified in the `pyproject.toml` file
and can be installed using `uv sync`.

(sec_development_python_style)=


### Code style

Python code in tskit is formatted and linted using
[ruff](https://docs.astral.sh/ruff/). These checks run automatically as part of
the {ref}`prek checks <sec_development_workflow_prek>` on each commit.

Ruff is quite opinionated and it gains more opinions on each version.
We therefore pin ruff to an exact version and maintain a list of "ignore"
classes in pyproject.toml. The version of ruff should be updated periodically
with fixes applied or the the list ignore extended as necessary.


(sec_development_python_tests)=


### Tests

The tests are defined in the `tests` directory, and run using
[pytest](https://docs.pytest.org/en/stable/) from the `python` directory.
If you want to run the tests in a particular module (say, `test_tables.py`), use:

```bash
$ uv run pytest tests/test_tables.py
```

To run all the tests in a particular class in this module (say, `TestNodeTable`)
use:

```bash
$ uv run pytest tests/test_tables.py::TestNodeTable
```

To run a specific test case in this class (say, `test_copy`) use:

```bash
$ uv run pytest tests/test_tables.py::TestNodeTable::test_copy
```

You can also run tests with a keyword expression search. For example this will
run all tests that have `TestNodeTable` but not `copy` in their name:

```bash
$ uv run pytest -k "TestNodeTable and not copy"
```

When developing your own tests, it is much quicker to run the specific tests
that you are developing rather than rerunning large sections of the test
suite each time.

To run all of the tests, we can use:

```bash
$ uv run pytest
```

By default the tests are run on 4 cores, if you have more you can specify:

```bash
$ uv run pytest -n8
```

A few of the tests take most of the time, we can skip the slow tests to get the test run
under 20 seconds on an modern workstation:

```bash
$ uv run pytest --skip-slow
```

If you have an agent running the tests in a sandboxed environment, you may need to
skip tests thsat require network access or FIFOs:

```bash
$ uv run pytest --skip-network
```

If you have a lot of failing tests it can be useful to have a shorter summary
of the failing lines:

```bash
$ uv run pytest --tb=line
```

If you need to see the output of tests (e.g. `print` statements) then you need to use
these flags to run a single thread and capture output:

```bash
$ uv run pytest -n0 -vs
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
The entry point `tskit_main` is declared under `[project.scripts]` in
`python/pyproject.toml`, which makes `tskit` available as a command after
installation.

The CLI can also be run using `uv run python -m tskit` during development.

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
On Debian/Ubuntu, install the system dependencies with:

```bash
$ sudo apt install libcunit1-dev ninja-build
```

Install meson using uv:

```bash
$ uv tool install meson
```

An exact version of clang-format is required because formatting rules
change from version to version. This is why we pin to an exact version
of clang-format in pyproject.toml, which gets used by prek linting.
If you wish to run clang-format yourself (e.g., within your editor)
a straightforward way to do this is to use ``uv tool install clang-format==[version]``,
which will install to your PATH.
However, you will need to manually keep track of what version is installed
(``uv tool list`` is useful for this).


(sec_development_c_code_style)=

### Code style

C code is formatted using
[clang-format](https://clang.llvm.org/docs/ClangFormat.html)
with a custom configuration. This is checked as part of the
{ref}`prek checks <sec_development_workflow_prek>`. To manually format all files run:

```bash
$ uv run prek --all-files
```

If you are doing this in the ``c`` directory, use
``uv run --project=../python prek --all-files``.


If you are getting obscure errors from prek, sometimes this is caused by
prek searching for configuration within subdirectories. To avoid this, tell
prek where to find its config explicitly:

```bash
$ uv run prek --all-files -c prek.toml
```


### Building

We use [meson](https://mesonbuild.com) and [ninja-build](https://ninja-build.org) to
compile the C code. Meson keeps all compiled binaries in a build directory (this has many advantages
such as allowing multiple builds with different options to coexist). The build configuration
is defined in `meson.build`. To set up the initial build
directory, run

```bash
$ cd c
$ meson setup build
```

To setup a debug build add `--buildtype=debug` to the above command. This will set the `TSK_TRACE_ERRORS`
flag, which will print error messages to `stderr` when errors occur which is useful for debugging.

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

### Compile flags

If the flag `TSK_TRACE_ERRORS` is defined (by e.g. `-DTSK_TRACE_ERRORS` to gcc),
then error messages will be printed to `stderr` when errors occur. This also allows
breakpoints to be set in the `_tsk_trace_error` function to break on all errors.

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


### Viewing coverage reports

To generate and view coverage reports for the C tests locally:

Compile with coverage enabled:
   ```bash
   $ cd c
   $ meson build -D b_coverage=true
   $ ninja -C build
   ```

Run the tests:
   ```bash
   $ ninja -C build test
   ```

Generate coverage data:
   ```bash
   $ cd build
   $ find ../tskit/*.c -type f -printf "%f\n" | xargs -i gcov -pb libtskit.a.p/tskit_{}.gcno ../tskit/{}
   ```

The generated `.gcov` files can then be viewed directly with `cat filename.c.gcov`.
Lines prefixed with `#####` were never executed, lines with numbers show execution counts, and lines with `-` are non-executable code.

`lcov` can be used to create browsable HTML coverage reports:
  ```bash
  $ sudo apt-get install lcov  # if needed
  $ lcov --capture --directory build-gcc --output-file coverage.info
  $ genhtml coverage.info --output-directory coverage_html
  $ firefox coverage_html/index.html
  ```

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
        ret = tsk_trace_error(TSK_ERR_NO_MEMORY);
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

When setting error codes in the C code, please use the `tsk_trace_error` function.
If `TSK_TRACE_ERRORS` is defined, this will print a message to stderr with the
details of the error.


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
module and how it is built from source. The module is built automatically by
`uv sync`, but if you modify the C extension code you will need to rebuild it.
The simplest way to do this is to run `make` in the `python` directory:

```bash
$ make
```

If `make` is not available, you can run the same command manually:

```bash
$ uv run python setup.py build_ext --inplace
```

It is sometimes useful to specify compiler flags when building the low
level module. For example, to make a debug build you can use:

```bash
$ CFLAGS='-Wall -O0 -g' make
```

If you need to track down a segfault etc, running some code through gdb can
be very useful. For example, to run a particular test case, we can do:


```bash
$ gdb python
(gdb) run -m pytest tests/test_python_c.py


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

Continuous integration is handled by [GitHub Actions](https://help.github.com/en/actions).
tskit uses shared workflows defined in the
[tskit-dev/.github](https://github.com/tskit-dev/.github) repository:

- **lint** — runs prek against all files
- **python-tests** — runs the pytest suite with coverage on Linux, macOS and Windows
- **python-c-tests** — builds the C extension with coverage and runs low-level tests
- **c-tests** — runs C unit tests under gcc, clang, and valgrind
- **docs** — builds the documentation and deploys it on merge to `main`
- **python-packaging** — validates the sdist and wheel

[CodeCov](https://codecov.io/gh) tracks test coverage for Python and C.


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
6.  Test your low-level implementation in [tskit/python/tests/test_python_c.py
   ](https://github.com/tskit-dev/tskit/blob/main/python/tests/test_python_c.py):
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

### prek is blocking me!

The prek hook is designed to make things easier, not harder. If the checks are
blocking you, feel free to skip them with `--no-verify` and sort it out before
the PR is merged. There’s no shame in a broken build.

```bash
> git commit -a -m ‘my changes’ --no-verify
```

### prek reports unexpected failures

If prek reports failures on files you didn’t edit, try clearing the cache:

```bash
> uv run prek cache clean
```

If that doesn’t help, you can reinstall the hook:

```bash
> uv run prek uninstall
> uv run prek install
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

See the [repo administration guide](https://github.com/tskit-dev/.github/blob/main/repo_administration.md)
for the release process. Tskit has both a C API release and a Python package release,
each covered in the tskit/kastore section of that document.

It is worth running the benchmarks (see above) before a Python release to check
for any unexpected major regressions. For a major release the website
(github repo tskit-dev/tskit-site) should be updated with a notebook of new
features and the `bench-results.html` updated.
