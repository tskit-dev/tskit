project('tskit', 'c')

cc = meson.get_compiler('c')
m_dep = cc.find_library('m', required : false)
cunit_dep = dependency('cunit')

extra_c_args = [
    '-std=c99', '-Wall', '-Wextra', '-Werror', '-Wpedantic', '-W',
    '-Wmissing-prototypes',  '-Wstrict-prototypes',
    '-Wconversion', '-Wshadow', '-Wpointer-arith', '-Wcast-align',
    '-Wcast-qual', '-Wwrite-strings', '-Wnested-externs',
    '-fshort-enums', '-fno-common']
lib_sources = [
    'kastore/c/kastore.c', 'tsk_core.c', 'tsk_tables.c', 'tsk_trees.c', 
    'tsk_genotypes.c', 'tsk_stats.c', 'tsk_convert.c']

kastore_dir=join_paths(meson.source_root(), 'kastore/c')

add_global_arguments(['-I' + kastore_dir], language: 'c')

tskit_lib = static_library('tskit', 
    sources: lib_sources, dependencies: m_dep, c_args: extra_c_args)

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


# The development CLI. Don't use extra C args because argtable code won't pass
executable('dev-cli', 
    sources: ['dev-tools/dev-cli.c', 'dev-tools/argtable3.c'], 
    link_with: [tskit_lib], c_args:['-Dlint'])


# Example programs. TODO make this optional to avoid falling over without GSL
gsl_dep = dependency('gsl')
executable('haploid_wright_fisher', 
    sources: ['examples/haploid_wright_fisher.c'], link_with: [tskit_lib],
    dependencies: [gsl_dep])

# TMP: until we've ported all the tests, keep this compilable.
# gsl_dep = dependency('gsl')
# executable('old_tests', sources: ['tests/old_tests.c'], link_with: tskit_lib, 
#     dependencies: [cunit_dep, gsl_dep])
