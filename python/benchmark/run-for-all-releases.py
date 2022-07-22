import json
import subprocess
from urllib.request import urlopen

import tqdm
from distutils.version import StrictVersion


def versions(package_name):
    url = f"https://pypi.org/pypi/{package_name}/json"
    data = json.load(urlopen(url))
    return sorted(data["releases"].keys(), key=StrictVersion)


def sh(command):
    subprocess.run(command, check=True, shell=True)


if __name__ == "__main__":
    try:
        sh("python -m venv _bench-temp-venv")
        sh("_bench-temp-venv/bin/pip install -r ../requirements/development.txt")
        versions = [
            v
            for v in versions("tskit")
            # We don't want alphas, betas or two broken versions:
            if "a" not in v and "b" not in v and v not in ("0.0.0", "0.1.0")
        ]
        for v in tqdm.tqdm(versions):
            sh(f"_bench-temp-venv/bin/pip install tskit=={v}")
            sh("_bench-temp-venv/bin/python run.py")
    finally:
        sh("rm -rf _bench-temp-venv")
