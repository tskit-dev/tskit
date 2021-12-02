import re
import sys

SUBS = [
    (r":user:`([A-Za-z0-9-]*)`", r"[@\1](https://github.com/\1)"),
    (r":pr:`([0-9]*)`", r"[#\1](https://github.com/tskit-dev/tskit/issues/\1)"),
    (r":issue:`([0-9]*)`", r"[#\1](https://github.com/tskit-dev/tskit/issues/\1)"),
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


with open(sys.argv[1]) as f:
    print("".join(process_log(f.readlines())))
