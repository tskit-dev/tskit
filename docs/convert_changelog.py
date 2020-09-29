import re
from pathlib import Path

SUBS = [
    (r":user:`([A-Za-z0-9-]*)`", r"[@\1](https://github.com/\1)"),
    (r":pr:`([0-9]*)`", r"[#\1](https://github.com/tskit-dev/tskit/issues/\1)"),
    (r":issue:`([0-9]*)`", r"[#\1](https://github.com/tskit-dev/tskit/issues/\1)"),
]

FILES = [
    Path(__file__).parent.parent / "c" / "CHANGELOG.rst",
    Path(__file__).parent.parent / "python" / "CHANGELOG.rst",
]


def process_log(log):
    delimiters_seen = 0
    for line in log:
        if line.startswith("-------"):
            delimiters_seen += 1
            continue
        if delimiters_seen == 3:
            return
        if delimiters_seen % 2 == 0:
            for pattern, replace in SUBS:
                line = re.sub(pattern, replace, line)
            yield line


for file in FILES:
    with open(file) as f:
        print("-------------")
        print(file)
        print("-------------")
        print("".join(process_log(f.readlines())))
