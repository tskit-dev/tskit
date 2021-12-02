project('example', 'c')

tskit_proj = subproject('tskit')
tskit_dep = tskit_proj.get_variable('tskit_dep')

executable('example',
  'example.c',
   dependencies : [tskit_dep],
   install : true)

