import os
import platform

import numpy
from setuptools import Extension
from setuptools import setup

IS_WINDOWS = platform.system() == "Windows"


libdir = "lib"
kastore_dir = os.path.join(libdir, "subprojects", "kastore")
tsk_source_files = [
    "core.c",
    "tables.c",
    "trees.c",
    "genotypes.c",
    "stats.c",
    "convert.c",
    "haplotype_matching.c",
]
sources = (
    ["_tskitmodule.c"]
    + [os.path.join(libdir, "tskit", f) for f in tsk_source_files]
    + [os.path.join(kastore_dir, "kastore.c")]
)

defines = []
libraries = []
if IS_WINDOWS:
    libraries.append("Advapi32")
    defines.append(("WIN32", None))

_tskit_module = Extension(
    "_tskit",
    sources=sources,
    extra_compile_args=["-std=c99"],
    libraries=libraries,
    define_macros=defines,
    include_dirs=["lwt_interface", libdir, kastore_dir, numpy.get_include()],
)

setup(
    ext_modules=[_tskit_module],
)
