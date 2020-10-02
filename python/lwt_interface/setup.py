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


libdir = "../lib"
kastore_dir = os.path.join(libdir, "subprojects", "kastore")
# TODO pathlib glob this.
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
    ["example_c_module.c"]
    + [os.path.join(libdir, "tskit", f) for f in tsk_source_files]
    + [os.path.join(kastore_dir, "kastore.c")]
)

defines = []
libraries = []
if IS_WINDOWS:
    # Needed for generating UUIDs in tskit
    libraries.append("Advapi32")
    defines.append(("WIN32", None))

extension_module = Extension(
    "example_c_module",
    sources=sources,
    extra_compile_args=["-std=c99"],
    libraries=libraries,
    define_macros=defines,
    include_dirs=[libdir, kastore_dir],
)

numpy_ver = "numpy>=1.7"

setup(
    name="example_c_module",
    description="Example usage of the LightweightTableCollection tskit interface",
    ext_modules=[extension_module],
    setup_requires=[numpy_ver],
    cmdclass={"build_ext": local_build_ext},
    license="MIT",
    platforms=["POSIX", "Windows", "MacOS X"],
)
