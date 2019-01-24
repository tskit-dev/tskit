project('tskit', ['c', 'cpp'], default_options: ['c_std=c99', 'cpp_std=c++11'])

kastore_proj = subproject('kastore')
kastore_dep = kastore_proj.get_variable('kastore_dep')

cc = meson.get_compiler('c')
m_dep = cc.find_library('m', required: false)
lib_deps = [m_dep, kastore_dep]

extra_c_args = [
    '-Wall', '-Wextra', '-Werror', '-Wpedantic', '-W',
    '-Wmissing-prototypes',  '-Wstrict-prototypes',
    '-Wconversion', '-Wshadow', '-Wpointer-arith', '-Wcast-align',
    '-Wcast-qual', '-Wwrite-strings', '-Wnested-externs',
    '-fshort-enums', '-fno-common']

lib_sources = [
    'tskit/core.c', 'tskit/tables.c', 'tskit/trees.c', 
    'tskit/genotypes.c', 'tskit/stats.c', 'tskit/convert.c']
lib_headers = [
    'tskit/core.h', 'tskit/tables.h', 'tskit/trees.h', 
    'tskit/genotypes.h', 'tskit/stats.h', 'tskit/convert.h']

# Subprojects use the static library for simplicity.
inc = include_directories(['.', 'tskit'])
tskit_lib = static_library('tskit', 
    sources: lib_sources, dependencies: lib_deps)
tskit_dep = declare_dependency(include_directories:inc, link_with: tskit_lib)

if not meson.is_subproject()

    # Shared library install target.
    shared_library('tskit', 
        sources: lib_sources, dependencies: lib_deps, c_args: extra_c_args, install: true)
    install_headers('tskit.h')
    install_headers(lib_headers, subdir: 'tskit') 

    cunit_dep = dependency('cunit')
    # We don't specify extra C args here as CUnit won't pass the checks.
    test_lib = static_library('testlib', 
        sources: ['tests/testlib.c'], dependencies: [cunit_dep, kastore_dep])

    test_core = executable('test_core', 
        sources: ['tests/test_core.c'], 
        link_with: [tskit_lib, test_lib], c_args: extra_c_args, dependencies: kastore_dep)
    test('core', test_core)

    test_tables = executable('test_tables', 
        sources: ['tests/test_tables.c'], 
        link_with: [tskit_lib, test_lib], c_args: extra_c_args, dependencies: kastore_dep)
    test('tables', test_tables)

    test_trees = executable('test_trees', 
        sources: ['tests/test_trees.c'], 
        link_with: [tskit_lib, test_lib], c_args: extra_c_args, dependencies: kastore_dep)
    test('trees', test_trees)

    test_genotypes = executable('test_genotypes', 
        sources: ['tests/test_genotypes.c'], 
        link_with: [tskit_lib, test_lib], c_args: extra_c_args, dependencies: kastore_dep)
    test('genotypes', test_genotypes)

    test_convert = executable('test_convert', 
        sources: ['tests/test_convert.c'], 
        link_with: [tskit_lib, test_lib], c_args: extra_c_args, dependencies: kastore_dep)
    test('convert', test_convert)
         
    test_stats = executable('test_stats', 
        sources: ['tests/test_stats.c'], 
        link_with: [tskit_lib, test_lib], c_args: extra_c_args, dependencies: kastore_dep)
    test('stats', test_stats)

    test_minimal_cpp = executable('test_minimal_cpp', 
        sources: ['tests/test_minimal_cpp.cpp'], link_with: [tskit_lib],
        dependencies: kastore_dep)
    test('minimal_cpp', test_minimal_cpp)

    # The development CLI. Don't use extra C args because argtable code won't pass
    executable('dev-cli', 
        sources: ['dev-tools/dev-cli.c', 'dev-tools/argtable3.c'], 
        link_with: [tskit_lib], c_args:['-Dlint'], dependencies: kastore_dep)

    # Example programs. 
    executable('api_structure', 
        sources: ['examples/api_structure.c'], link_with: [tskit_lib], dependencies: lib_deps)
    executable('error_handling', 
        sources: ['examples/error_handling.c'], link_with: [tskit_lib], dependencies: lib_deps)
    executable('tree_iteration', 
        sources: ['examples/tree_iteration.c'], link_with: [tskit_lib], dependencies: lib_deps)

    gsl_dep = dependency('gsl', required: false)
    if gsl_dep.found()
        executable('haploid_wright_fisher', 
            sources: ['examples/haploid_wright_fisher.c'], link_with: [tskit_lib],
            dependencies: [gsl_dep, lib_deps])
    endif
endif

# TMP: until we've ported all the tests, keep this compilable.
# gsl_dep = dependency('gsl')
# executable('old_tests', sources: ['tests/old_tests.c'], link_with: tskit_lib, 
#     dependencies: [cunit_dep, gsl_dep])
