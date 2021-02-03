# MIT License
#
# Copyright (c) 2018-2020 Tskit Developers
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
Configuration and fixtures for pytest. Only put test-suite wide fixtures in here. Module
specific fixtures should live in their modules.

To use a fixture in a test simply refer to it by name as an argument. This is called
dependancy injection. Note that all fixtures should have the suffix "_fixture" to make
it clear in test code.

For example to use the `ts` fixture (a tree sequence with data in all fields) in a test:

class TestClass:
    def test_something(self, ts_fixture):
        assert ts_fixture.some_method() == expected

Fixtures can be parameterised etc. see https://docs.pytest.org/en/stable/fixture.html

Note that fixtures have a "scope" for example `ts_fixture` below is only created once
per test session and re-used for subsequent tests.
"""
import msprime
import pytest
from pytest import fixture

from . import ts_examples


def pytest_addoption(parser):
    """
    Add an option to skip tests marked with `@pytest.mark.slow`
    """
    parser.addoption(
        "--skip-slow", action="store_true", default=False, help="Skip slow tests"
    )


def pytest_configure(config):
    """
    Add docs on the "slow" marker
    """
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--skip-slow"):
        skip_slow = pytest.mark.skip(reason="--skip-slow specified")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)


# NB: parameters from the fixture below can be overridden with any function names
# in "ts_examples.py", or any list, e.g. params=ts_examples.internal_samples
@fixture(scope="session", params=["complete_ts"])  # By default return a complete ts
def ts_fixture(request):
    return getattr(ts_examples, request.param)()


@fixture(scope="session")
def replicate_ts_fixture():
    """
    A list of tree sequences
    """
    return list(msprime.simulate(10, num_replicates=10, random_seed=42))
