"""
Code to stress the low-level API as much as possible to expose
any memory leaks or error handling issues.
"""
import argparse
import curses
import logging
import os
import random
import resource
import time
import tracemalloc
import unittest

import tests.test_dict_encoding as test_dict_encoding
import tests.test_file_format as test_file_format
import tests.test_haplotype_matching as test_haplotype_matching
import tests.test_highlevel as test_highlevel
import tests.test_lowlevel as test_lowlevel
import tests.test_metadata as test_metadata
import tests.test_stats as test_stats
import tests.test_tables as test_tables
import tests.test_threads as test_threads
import tests.test_topology as test_topology
import tests.test_tree_stats as test_tree_stats
import tests.test_vcf as test_vcf


def main(stdscr):
    modules = {
        "highlevel": test_highlevel,
        "lowlevel": test_lowlevel,
        "vcf": test_vcf,
        "threads": test_threads,
        "stats": test_stats,
        "tree_stats": test_tree_stats,
        "tables": test_tables,
        "file_format": test_file_format,
        "topology": test_topology,
        "dict_encoding": test_dict_encoding,
        "haplotype_matching": test_haplotype_matching,
        "metadata": test_metadata,
    }
    parser = argparse.ArgumentParser(
        description="Run tests in a loop to stress low-level interface"
    )
    parser.add_argument(
        "-m",
        "--module",
        help="Run tests only on this module",
        choices=list(modules.keys()),
    )
    args = parser.parse_args()
    test_modules = list(modules.values())
    if args.module is not None:
        test_modules = [modules[args.module]]

    # Need to do this to silence the errors from the file_format tests.
    logging.basicConfig(level=logging.ERROR)

    max_rss = 0
    max_rss_iter = 0
    min_rss = 1e100
    iteration = 0
    last_print = time.time()
    devnull = open(os.devnull, "w")
    tracemalloc.start()
    memory_start = None
    while True:
        # We don't want any random variation in the amount of memory
        # used from test-to-test.
        random.seed(1)
        testloader = unittest.TestLoader()
        suite = testloader.loadTestsFromModule(test_modules[0])
        for mod in test_modules[1:]:
            suite.addTests(testloader.loadTestsFromModule(mod))
        runner = unittest.TextTestRunner(verbosity=0, stream=devnull)
        if memory_start is None:
            memory_start = tracemalloc.take_snapshot()
        result = runner.run(suite)
        memory_current = tracemalloc.take_snapshot()
        rusage = resource.getrusage(resource.RUSAGE_SELF)
        if max_rss < rusage.ru_maxrss:
            max_rss = rusage.ru_maxrss
            max_rss_iter = iteration
        if min_rss > rusage.ru_maxrss:
            min_rss = rusage.ru_maxrss

        # We don't want to flood stdout, so we rate-limit to 1 per second.
        if time.time() - last_print > 1:
            stdscr.clear()
            rows, cols = stdscr.getmaxyx()
            stdscr.addstr(
                0,
                0,
                "iter\ttests\terr\tfail\tskip\tRSS\tmin\tmax\tmax@iter"[: cols - 1],
            )
            stdscr.addstr(
                1,
                0,
                "\t".join(
                    map(
                        str,
                        [
                            iteration,
                            result.testsRun,
                            len(result.failures),
                            len(result.errors),
                            len(result.skipped),
                            rusage.ru_maxrss,
                            min_rss,
                            max_rss,
                            max_rss_iter,
                        ],
                    )
                )[: cols - 1],
            )
            stats = memory_current.compare_to(memory_start, "traceback")
            for i, stat in enumerate(stats[: rows - 3], 1):
                stdscr.addstr(i + 2, 0, str(stat)[: cols - 1])
            last_print = time.time()
            stdscr.refresh()

        iteration += 1


if __name__ == "__main__":
    stdscr = curses.initscr()
    curses.noecho()
    curses.cbreak()

    try:
        main(stdscr)
    finally:
        curses.echo()
        curses.nocbreak()
        curses.endwin()
