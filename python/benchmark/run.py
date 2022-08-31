import json
import os.path
import platform
import sys
import timeit
from pathlib import Path

import click
import psutil
import tqdm
import yaml
from matplotlib.colors import LinearSegmentedColormap
from si_prefix import si_format

tskit_dir = Path(__file__).parent.parent
sys.path.append(str(tskit_dir))
import tskit  # noqa: E402
import msprime  # noqa: E402

with open("config.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


def system_info():
    ret = {}
    uname = platform.uname()
    for attr in ["system", "node", "release", "version", "machine", "processor"]:
        ret[attr] = getattr(uname, attr)
    ret["python_version"] = sys.version
    cpufreq = psutil.cpu_freq()
    ret["physical_cores"] = psutil.cpu_count(logical=False)
    ret["total_cores"] = psutil.cpu_count(logical=True)
    ret["max_frequency"] = cpufreq.max
    ret["min_frequency"] = cpufreq.min
    ret["current_frequency"] = cpufreq.current
    ret["cpu_usage_per_core"] = [
        percentage for percentage in psutil.cpu_percent(percpu=True, interval=1)
    ]
    ret["total_cpu_usage"] = psutil.cpu_percent()
    return ret


def make_file():
    benchmark_trees = tskit_dir / "benchmark" / "bench.trees"
    if not os.path.exists(benchmark_trees):
        print("Generating benchmark trees...")
        demography = msprime.Demography()
        demography.add_population(name="A", initial_size=10_000)
        demography.add_population(name="B", initial_size=5_000)
        demography.add_population(name="C", initial_size=1_000)
        demography.add_population_split(time=1000, derived=["A", "B"], ancestral="C")
        ts = msprime.sim_ancestry(
            samples={"A": 25000, "B": 25000},
            demography=demography,
            sequence_length=1_000_000,
            random_seed=42,
            recombination_rate=0.0000001,
            record_migrations=True,
            record_provenance=True,
        )
        ts = msprime.sim_mutations(ts, rate=0.000001, random_seed=42)
        ts.dump(benchmark_trees)
        ts = msprime.sim_ancestry(
            samples={"A": 1, "B": 1},
            demography=demography,
            sequence_length=1,
            random_seed=42,
            recombination_rate=0,
            record_migrations=True,
            record_provenance=True,
        )
        ts = msprime.sim_mutations(ts, rate=0.001, random_seed=42)
        ts.dump(tskit_dir / "benchmark" / "tiny.trees")


def autotime(setup, code):
    t = timeit.Timer(setup=setup, stmt=code)
    try:
        one_run = t.timeit(number=1)
    except Exception as e:
        print(f"{code}: Error running benchmark: {e}")
        return None
    num_trials = int(max(1, 2 / one_run))
    return one_run, num_trials, t.timeit(number=num_trials) / num_trials


def run_benchmarks(keyword_filter):
    results = {}
    for benchmark in tqdm.tqdm(config["benchmarks"]):
        bench_name = benchmark.get("name", benchmark["code"])
        if keyword_filter not in bench_name:
            continue
        params = benchmark.get("parameters", {"noop": [None]})

        # Expand the parameters
        def sub_expand(context, name, d):
            if isinstance(d, dict):
                ret = []
                for k, v in d.items():
                    new_context = {**{k: v for k, v in context.items()}, name: k}
                    for k2, v2 in v.items():
                        ret += sub_expand(new_context, k2, v2)
                return ret
            elif isinstance(d, list):
                return [
                    {**{k: v for k, v in context.items()}, name: value} for value in d
                ]
            else:
                raise ValueError(f"Invalid parameter type: {type(d)}-{d}")

        expanded_params = []
        for k, v in params.items():
            expanded_params += sub_expand({}, k, v)

        for values in expanded_params:
            setup = (
                f"import sys;sys.path.append('{tskit_dir}');"
                + config["setup"].replace("\n", "\n")
                + benchmark.get("setup", "").replace("\n", "\n").format(**values)
            )
            code = benchmark["code"].replace("\n", "\n").format(**values)
            result = autotime(setup, code)
            if result is not None:
                one_run, num_trials, avg = result
                results.setdefault(bench_name, {})[code] = {
                    "one_run": one_run,
                    "num_trials": num_trials,
                    "avg": avg,
                }

    return results


def generate_report(all_versions_results):
    all_benchmarks = {}
    for _version, results in all_versions_results.items():
        for benchmark, values in results["tskit_benchmarks"].items():
            for code in values.keys():
                all_benchmarks.setdefault(benchmark, set()).add(code)

    all_versions = sorted(all_versions_results.keys())

    cmap = LinearSegmentedColormap.from_list("rg", ["g", "w", "r"], N=256)

    with open(tskit_dir / "benchmark" / "bench-results.html", "w") as f:
        f.write("<html><body>\n")
        f.write("<h1>tskit benchmark results</h1>\n")
        f.write("<table>\n")
        f.write("<tr><th></th>")
        for version in all_versions:
            f.write(f"<th>{version}</th>")
        f.write("</tr>\n")
        for benchmark in sorted(all_benchmarks.keys()):
            values = all_benchmarks[benchmark]
            indent = False
            if len(values) > 1:
                indent = True
                f.write(
                    f"<tr>"
                    f"  <td style='font-family: monospace'>"
                    f"    {benchmark}"
                    f"  </td>"
                    f"</tr>\n"
                )
            for code in sorted(values):
                f.write(
                    f"<tr><td style='font-family: monospace;"
                    f"padding-left: {'10px' if indent else 'inherit'}'>{code}</td>"
                )
                last_avg = None
                for version in all_versions:
                    try:
                        avg = all_versions_results[version]["tskit_benchmarks"][
                            benchmark
                        ][code]["avg"]
                        if last_avg is not None:
                            percent_change = 100 * ((avg - last_avg) / last_avg)
                            col = cmap(int(((percent_change / 100) * 128) + 128))
                            f.write(
                                f"<td style='background-color: rgba({col[0]*255},"
                                f" {col[1]*255}, {col[2]*255}, 1)'>"
                            )

                            f.write(f"{si_format(avg)} ({percent_change:.1f}%)")
                        else:
                            f.write(f"<td>{si_format(avg)}</td>")
                        last_avg = avg
                    except KeyError:
                        f.write("<td>N/A</td>")

                f.write("</tr>\n")
        f.write("</table>\n")


def print_result(results):
    max_name_length = max(len(name) for name in results.keys()) + 1
    for _bench, param_results in results.items():
        for name, data in param_results.items():
            print(name.ljust(max_name_length), si_format(data["avg"]))


@click.command()
@click.option(
    "--keyword_filter",
    "-k",
    type=str,
    default="",
    help="Only benchmarks with a name containing this string will be run",
)
@click.option("--print_results", "-p", is_flag=True, help="Print results to STDOUT")
def run_benchmark_and_save(keyword_filter, print_results):
    print("Benchmarking tskit version:", tskit._version.tskit_version)
    make_file()
    results = {}
    results["system"] = system_info()
    results["tskit_benchmarks"] = run_benchmarks(keyword_filter)

    if print_results:
        print_result(results["tskit_benchmarks"])

    all_versions_results = {}
    results_json = tskit_dir / "benchmark" / "bench-results.json"
    if os.path.exists(results_json):
        with open(results_json) as f:
            all_versions_results = json.load(f)

    all_versions_results[tskit._version.tskit_version] = results
    with open(results_json, "w") as f:
        json.dump(all_versions_results, f, indent=2)
    generate_report(all_versions_results)

    sys.exit(0)


if __name__ == "__main__":
    run_benchmark_and_save()
