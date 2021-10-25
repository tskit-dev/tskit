import os.path
import platform

from setuptools import Extension
from setuptools import setup
from setuptools.command.build_ext import build_ext


IS_WINDOWS = platform.system() == "Windows"


# Obscure magic required to allow numpy be used as a 'setup_requires'.
# Based on https://stackoverflow.com/questions/19919905
class local_build_ext(build_ext):
    def finalize_options(self):
        build_ext.finalize_options(self)
        import builtins

        # Prevent numpy from thinking it is still in its setup process:
        builtins.__NUMPY_SETUP__ = False
        import numpy

        self.include_dirs.append(numpy.get_include())


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
    # Needed for generating UUIDs
    libraries.append("Advapi32")
    defines.append(("WIN32", None))

_tskit_module = Extension(
    "_tskit",
    sources=sources,
    extra_compile_args=["-std=c99"],
    libraries=libraries,
    define_macros=defines,
    include_dirs=["lwt_interface", libdir, kastore_dir],
)


# After exec'ing this file we have tskit_version defined.
tskit_version = None  # Keep PEP8 happy.
version_file = os.path.join("tskit", "_version.py")
with open(version_file) as f:
    exec(f.read())

setup(
    # The package name along with all the other metadata is specified in setup.cfg
    # However, GitHub's dependency graph can't see the package unless we put this here.
    name="tskit",
    version=tskit_version,
    ext_modules=[_tskit_module],
    cmdclass={"build_ext": local_build_ext},
)
