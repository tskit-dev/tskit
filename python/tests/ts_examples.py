"""
A file containing public functions that each return an example tree sequence, for use
in test fixtures. The basic fixture, defined in `conftest.py` is called `example_ts`,
so to use a single complete ts in a test function, simply include `example_ts` as a param

    test_myfunc(self, example_ts):
        assert example_ts.num_samples > 0

To use another, or a set of example tree sequences, you can use @pytest.mark.parametrize

    @pytest.mark.parametrize("example_ts", ["simple_ts", "complete_ts"])
    test_myfunc(self, example_ts):
        assert example_ts.num_samples > 0

This file also provides lists of example tree sequences of a particular type:

    from . import ts_examples
    @pytest.mark.parametrize("example_ts", ts_examples.internal_samples)
    test_myfunc(self, example_ts):
        assert example_ts.num_samples > 0
"""
import msprime

import tskit


def simple_ts():
    return msprime.simulate(10, random_seed=42)


def complete_ts():
    """
    A tree sequence with data in all fields
    """
    n = 10
    t = 1
    population_configurations = [
        msprime.PopulationConfiguration(n // 2),
        msprime.PopulationConfiguration(n // 2),
        msprime.PopulationConfiguration(0),
    ]
    demographic_events = [
        msprime.MassMigration(time=t, source=0, destination=2),
        msprime.MassMigration(time=t, source=1, destination=2),
    ]
    ts = msprime.simulate(
        population_configurations=population_configurations,
        demographic_events=demographic_events,
        random_seed=1,
        mutation_rate=1,
        record_migrations=True,
    )
    tables = ts.dump_tables()
    # TODO replace this with properly linked up individuals using sim_ancestry
    # once 1.0 is released.
    for j in range(n):
        tables.individuals.add_row(flags=j, location=(j, j), parents=(j, j))

    for name, table in tables.name_map.items():
        if name != "provenances":
            table.metadata_schema = tskit.MetadataSchema({"codec": "json"})
            metadatas = [f"n_{name}_{u}" for u in range(len(table))]
            metadata, metadata_offset = tskit.pack_strings(metadatas)
            table.set_columns(
                **{
                    **table.asdict(),
                    "metadata": metadata,
                    "metadata_offset": metadata_offset,
                }
            )
    tables.metadata_schema = tskit.MetadataSchema({"codec": "json"})
    tables.metadata = "Test metadata"
    return tables.tree_sequence()


# 3 methods to return example tree sequences with internal samples:
# (copied from test_highlevel.py)
def all_nodes_samples():
    n = 5
    ts = msprime.simulate(n, random_seed=10, mutation_rate=5)
    assert ts.num_mutations > 0
    tables = ts.dump_tables()
    nodes = tables.nodes
    flags = nodes.flags
    # Set all nodes to be samples.
    flags[:] = tskit.NODE_IS_SAMPLE
    nodes.flags = flags
    return tables.tree_sequence()


def only_internal_samples():
    n = 5
    ts = msprime.simulate(n, random_seed=10, mutation_rate=5)
    assert ts.num_mutations > 0
    tables = ts.dump_tables()
    nodes = tables.nodes
    flags = nodes.flags
    # Set just internal nodes to be samples.
    flags[:] = 0
    flags[n:] = tskit.NODE_IS_SAMPLE
    nodes.flags = flags
    return tables.tree_sequence()


def mixed_node_samples():
    n = 5
    ts = msprime.simulate(n, random_seed=10, mutation_rate=5)
    assert ts.num_mutations > 0
    tables = ts.dump_tables()
    nodes = tables.nodes
    flags = nodes.flags
    # Set a mixture of internal and leaf samples.
    flags[:] = 0
    flags[n // 2 : n + n // 2] = tskit.NODE_IS_SAMPLE
    nodes.flags = flags
    return tables.tree_sequence()


internal_samples = ["all_nodes_samples", "only_internal_samples", "mixed_node_samples"]
