import glob
import os

import numpy as np
from Cython.Build import cythonize
from setuptools import setup
from setuptools.extension import Extension

TSKIT_BASE = os.path.join(os.path.dirname(__file__), "..", "..", "..")
TSKIT_C_PATH = os.path.join(TSKIT_BASE, "c")
TSKIT_PY_PATH = os.path.join(TSKIT_BASE, "python/lwt_interface")
KASTORE_PATH = os.path.join(TSKIT_BASE, "c", "subprojects", "kastore")
include_dirs = [TSKIT_C_PATH, TSKIT_PY_PATH, KASTORE_PATH, np.get_include()]

tskit_sourcefiles = list(glob.glob(os.path.join(TSKIT_C_PATH, "tskit", "*.c"))) + [
    os.path.join(KASTORE_PATH, "kastore.c")
]

extensions = [
    Extension(
        "_lwtc",
        ["_lwtc.c"] + tskit_sourcefiles,
        language="c",
        include_dirs=include_dirs,
    ),
    Extension(
        "example",
        ["example.pyx"] + tskit_sourcefiles,
        language="c",
        include_dirs=include_dirs,
    ),
]

extensions = cythonize(extensions, language_level=3)

setup(
    name="tskit_cython_example",
    version="0.0.1",
    ext_modules=extensions,
)
