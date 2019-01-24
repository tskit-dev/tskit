project('example', 'c')

kastore_proj = subproject('kastore')
kastore_dep = kastore_proj.get_variable('kastore_dep')
tskit_proj = subproject('tskit')
tskit_dep = tskit_proj.get_variable('tskit_dep')

executable('example',
  'example.c',
   dependencies : [kastore_dep, tskit_dep],
   install : true)

