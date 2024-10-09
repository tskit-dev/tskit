# MIT License
#
# Copyright (c) 2018-2024 Tskit Developers
# Copyright (C) 2018 University of Oxford
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
Tests for the provenance information attached to tree sequences.
"""
import json
import os
import platform
import sys
import time

try:
    import resource
except ImportError:
    resource = None  # resource absent on windows


import msprime
import pytest

import _tskit
import tskit
import tskit.provenance as provenance


def get_provenance(
    software_name="x",
    software_version="y",
    schema_version="1",
    environment=None,
    parameters=None,
):
    """
    Utility function to return a provenance document for testing.
    """
    document = {
        "schema_version": schema_version,
        "software": {"name": software_name, "version": software_version},
        "environment": {} if environment is None else environment,
        "parameters": {} if parameters is None else parameters,
    }
    return document


class TestSchema:
    """
    Tests for schema validation.
    """

    def test_empty(self):
        with pytest.raises(tskit.ProvenanceValidationError):
            tskit.validate_provenance({})

    def test_missing_keys(self):
        minimal = get_provenance()
        tskit.validate_provenance(minimal)
        for key in minimal.keys():
            copy = dict(minimal)
            del copy[key]
            with pytest.raises(tskit.ProvenanceValidationError):
                tskit.validate_provenance(copy)
        copy = dict(minimal)
        del copy["software"]["name"]
        with pytest.raises(tskit.ProvenanceValidationError):
            tskit.validate_provenance(copy)
        copy = dict(minimal)
        del copy["software"]["version"]
        with pytest.raises(tskit.ProvenanceValidationError):
            tskit.validate_provenance(copy)

    def test_software_types(self):
        for bad_type in [0, [1, 2, 3], {}]:
            doc = get_provenance(software_name=bad_type)
            with pytest.raises(tskit.ProvenanceValidationError):
                tskit.validate_provenance(doc)
            doc = get_provenance(software_version=bad_type)
            with pytest.raises(tskit.ProvenanceValidationError):
                tskit.validate_provenance(doc)

    def test_schema_version_empth(self):
        doc = get_provenance(schema_version="")
        with pytest.raises(tskit.ProvenanceValidationError):
            tskit.validate_provenance(doc)

    def test_software_empty_strings(self):
        doc = get_provenance(software_name="")
        with pytest.raises(tskit.ProvenanceValidationError):
            tskit.validate_provenance(doc)
        doc = get_provenance(software_version="")
        with pytest.raises(tskit.ProvenanceValidationError):
            tskit.validate_provenance(doc)

    def test_minimal(self):
        minimal = {
            "schema_version": "1",
            "software": {"name": "x", "version": "y"},
            "environment": {},
            "parameters": {},
        }
        tskit.validate_provenance(minimal)

    def test_extra_stuff(self):
        extra = {
            "you": "can",
            "schema_version": "1",
            "software": {"put": "anything", "name": "x", "version": "y"},
            "environment": {"extra": ["you", "want"]},
            "parameters": {"so": ["long", "its", "JSON", 0]},
        }
        tskit.validate_provenance(extra)

    def test_resources(self):
        resources = {
            "schema_version": "1",
            "software": {"name": "x", "version": "y"},
            "environment": {},
            "parameters": {},
            "resources": {
                "elapsed_time": 1,
                "user_time": 2,
                "sys_time": 3,
                "max_memory": 4,
            },
        }
        tskit.validate_provenance(resources)

    def test_resources_error(self):
        resources = {
            "schema_version": "1",
            "software": {"name": "x", "version": "y"},
            "environment": {},
            "parameters": {},
            "resources": {
                "elapsed_time": "1",
                "user_time": 2,
                "sys_time": 3,
                "max_memory": 4,
            },
        }
        with pytest.raises(tskit.ProvenanceValidationError):
            tskit.validate_provenance(resources)


class TestOutputProvenance:
    """
    Check that the schemas we produce in tskit are valid.
    """

    def test_simplify(self):
        ts = msprime.simulate(5, random_seed=1)
        ts = ts.simplify()
        prov = json.loads(ts.provenance(1).record)
        tskit.validate_provenance(prov)
        assert prov["parameters"]["command"] == "simplify"
        assert prov["environment"] == provenance.get_environment(include_tskit=False)
        assert prov["software"] == {"name": "tskit", "version": tskit.__version__}


class TestEnvironment:
    """
    Tests for the environment provenance.
    """

    def test_os(self):
        env = provenance.get_environment()
        os = {
            "system": platform.system(),
            "node": platform.node(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        }
        assert env["os"] == os

    def test_python(self):
        env = provenance.get_environment()
        python = {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        }
        assert env["python"] == python

    def test_libraries(self):
        kastore_lib = {"version": ".".join(map(str, _tskit.get_kastore_version()))}
        env = provenance.get_environment()
        assert {"kastore": kastore_lib, "tskit": {"version": tskit.__version__}} == env[
            "libraries"
        ]

        env = provenance.get_environment(include_tskit=False)
        assert {"kastore": kastore_lib} == env["libraries"]

        extra_libs = {"abc": [], "xyz": {"one": 1}}
        env = provenance.get_environment(include_tskit=False, extra_libs=extra_libs)
        libs = {"kastore": kastore_lib}
        libs.update(extra_libs)
        assert libs == env["libraries"]


class TestGetResources:
    def test_get_resources_keys(self):
        resources = provenance.get_resources(0)
        assert "elapsed_time" in resources
        assert "user_time" in resources
        assert "sys_time" in resources
        if resource is not None:
            assert "max_memory" in resources

    def test_get_resources_values(self):
        delta = 0.1
        t = time.time()
        resources = provenance.get_resources(t - delta)
        assert isinstance(resources["elapsed_time"], float)
        assert isinstance(resources["user_time"], float)
        assert isinstance(resources["sys_time"], float)
        assert resources["elapsed_time"] >= delta - 0.001
        assert resources["user_time"] > 0
        assert resources["sys_time"] > 0
        if resource is not None:
            assert isinstance(resources["max_memory"], int)
            assert resources["max_memory"] > 1024

    def test_get_resources_platform(self):
        resources = provenance.get_resources(0)
        if sys.platform != "darwin" and resource is not None:
            assert resources["max_memory"] % 1024 == 0


class TestGetSchema:
    """
    Ensure we return the correct JSON schema.
    """

    def test_file_equal(self):
        s1 = provenance.get_schema()
        base = os.path.join(os.path.dirname(__file__), "..", "tskit")
        with open(os.path.join(base, "provenance.schema.json")) as f:
            s2 = json.load(f)
        assert s1 == s2

    def test_caching(self):
        n = 10
        schemas = [provenance.get_schema() for _ in range(n)]
        # Ensure all the schemas are different objects.
        assert len(set(map(id, schemas))) == n
        # Ensure the schemas are all equal
        for j in range(n):
            assert schemas[0] == schemas[j]

    def test_form(self):
        s = provenance.get_schema()
        assert s["schema"] == "http://json-schema.org/draft-07/schema#"
        assert s["version"] == "1.1.0"


class TestTreeSeqEditMethods:
    """
    Ensure that tree sequence 'edit' methods correctly record themselves
    """

    def test_keep_delete_different(self):
        ts = msprime.simulate(5, random_seed=1)
        ts_keep = ts.keep_intervals([[0.25, 0.5]])
        ts_del = ts.delete_intervals([[0, 0.25], [0.5, 1.0]])
        assert ts_keep.num_provenances == ts_del.num_provenances
        for i, (p1, p2) in enumerate(zip(ts_keep.provenances(), ts_del.provenances())):
            if i == ts_keep.num_provenances - 1:
                # last one should be different
                assert p1.record != p2.record
            else:
                assert p1.record == p2.record
