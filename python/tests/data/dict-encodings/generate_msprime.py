import pathlib
import pickle

import _msprime
import msprime

pop_configs = [msprime.PopulationConfiguration(5) for _ in range(2)]
migration_matrix = [[0, 1], [1, 0]]
ts = msprime.simulate(
    population_configurations=pop_configs,
    migration_matrix=migration_matrix,
    mutation_rate=1,
    record_migrations=True,
    random_seed=1,
)
lwt = _msprime.LightweightTableCollection()
lwt.fromdict(ts.tables.asdict())

test_dir = pathlib.Path(__file__).parent
with open(test_dir / f"msprime-{msprime.__version__}.pkl", "wb") as f:
    pickle.dump(lwt.asdict(), f)
