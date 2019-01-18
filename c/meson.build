project('tskit', 'c', default_options: ['c_std=c99'])

kastore_dir=join_paths(meson.source_root(), 'kastore/c')
add_global_arguments(['-I' + kastore_dir], language: 'c')

cc = meson.get_compiler('c')
m_dep = cc.find_library('m', required: false)
lib_deps = [m_dep]

extra_c_args = [
    '-Wall', '-Wextra', '-Werror', '-Wpedantic', '-W',
    '-Wmissing-prototypes',  '-Wstrict-prototypes',
    '-Wconversion', '-Wshadow', '-Wpointer-arith', '-Wcast-align',
    '-Wcast-qual', '-Wwrite-strings', '-Wnested-externs',
    '-fshort-enums', '-fno-common']

lib_sources = [
    'kastore/c/kastore.c', 'tskit/core.c', 'tskit/tables.c', 'tskit/trees.c', 
    'tskit/genotypes.c', 'tskit/stats.c', 'tskit/convert.c']
lib_headers = [
    'tskit/core.h', 'tskit/tables.h', 'tskit/trees.h', 
    'tskit/genotypes.h', 'tskit/stats.h', 'tskit/convert.h']


# Shared library install target.
shared_library('tskit', 
    sources: lib_sources, dependencies: lib_deps, c_args: extra_c_args, install: true)
# FIXME for now we include the kastore header because the shared lib won't 
# work otherwise. Will need to come up with a better solution for this.
install_headers('kastore/c/kastore.h')
install_headers('tskit.h')
install_headers(lib_headers, subdir: 'tskit') 

# Static library for building tests and examples.
tskit_lib = static_library('tskit', 
    sources: lib_sources, dependencies: lib_deps, c_args: extra_c_args)

cunit_dep = dependency('cunit', required: false)
if cunit_dep.found()
    # We don't specify extra C args here as CUnit won't pass the checks.
    test_lib = static_library('testlib', 
        sources: ['tests/testlib.c'], dependencies: cunit_dep)

    test_core = executable('test_core', 
        sources: ['tests/test_core.c'], 
        link_with: [tskit_lib, test_lib], c_args: extra_c_args)
    test('core', test_core)

    test_tables = executable('test_tables', 
        sources: ['tests/test_tables.c'], 
        link_with: [tskit_lib, test_lib], c_args: extra_c_args)
    test('tables', test_tables)

    test_trees = executable('test_trees', 
        sources: ['tests/test_trees.c'], 
        link_with: [tskit_lib, test_lib], c_args: extra_c_args)
    test('trees', test_trees)

    test_genotypes = executable('test_genotypes', 
        sources: ['tests/test_genotypes.c'], 
        link_with: [tskit_lib, test_lib], c_args: extra_c_args)
    test('genotypes', test_genotypes)

    test_convert = executable('test_convert', 
        sources: ['tests/test_convert.c'], 
        link_with: [tskit_lib, test_lib], c_args: extra_c_args)
    test('convert', test_convert)
         
    test_stats = executable('test_stats', 
        sources: ['tests/test_stats.c'], 
        link_with: [tskit_lib, test_lib], c_args: extra_c_args)
    test('stats', test_stats)
endif


# The development CLI. Don't use extra C args because argtable code won't pass
executable('dev-cli', 
    sources: ['dev-tools/dev-cli.c', 'dev-tools/argtable3.c'], 
    link_with: [tskit_lib], c_args:['-Dlint'])

# Example programs. 
executable('api_structure', 
    sources: ['examples/api_structure.c'], link_with: [tskit_lib])
executable('error_handling', 
    sources: ['examples/error_handling.c'], link_with: [tskit_lib])
executable('tree_iteration', 
    sources: ['examples/tree_iteration.c'], link_with: [tskit_lib])

gsl_dep = dependency('gsl', required: false)
if gsl_dep.found()
    executable('haploid_wright_fisher', 
        sources: ['examples/haploid_wright_fisher.c'], link_with: [tskit_lib],
        dependencies: [gsl_dep])
endif

# TMP: until we've ported all the tests, keep this compilable.
# gsl_dep = dependency('gsl')
# executable('old_tests', sources: ['tests/old_tests.c'], link_with: tskit_lib, 
#     dependencies: [cunit_dep, gsl_dep])
