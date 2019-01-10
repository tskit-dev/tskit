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
    sources: ['testlib.c'], dependencies: cunit_dep)

executable('test_core', 
    sources: ['test_core.c'], 
    link_with: [tskit_lib, test_lib], c_args: extra_c_args)
executable('test_tables', 
    sources: ['test_tables.c'], 
    link_with: [tskit_lib, test_lib], c_args: extra_c_args)
executable('test_trees', 
    sources: ['test_trees.c'], 
    link_with: [tskit_lib, test_lib], c_args: extra_c_args)
executable('test_genotypes', 
    sources: ['test_genotypes.c'], 
    link_with: [tskit_lib, test_lib], c_args: extra_c_args)
