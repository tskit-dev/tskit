---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.12
    jupytext_version: 1.9.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

:::{currentmodule} tskit
:::


(sec_glossary)=

# Glossary

(sec_data_model_definitions)=

## Definitions

Here are some definitions of some key ideas encountered in this documentation.

(sec_data_model_definitions_tree)=

tree
: A "gene tree", i.e., the genealogical tree describing how a collection of
  genomes (usually at the tips of the tree) are related to each other at some
  chromosomal location. See {ref}`sec_nodes_or_individuals` for discussion
  of what a "genome" is.

(sec_data_model_definitions_tree_sequence)=

tree sequence
: A "succinct tree sequence" (or tree sequence, for brevity) is an efficient
  encoding of a sequence of correlated trees, such as one encounters looking
  at the gene trees along a genome. A tree sequence efficiently captures the
  structure shared by adjacent trees, (essentially) storing only what differs
  between them.

(sec_data_model_definitions_node)=

node
: Each branching point in each tree is associated with a particular genome
  in a particular ancestor, called a "node".  Since each node represents a
  specific genome it has a unique `time`, thought of as its birth time,
  which determines the height of any branching points it is associated with.
  See {ref}`sec_nodes_or_individuals` for discussion of what a "node" is.

(sec_data_model_definitions_individual)=

individual
: In certain situations we are interested in how nodes (representing
  individual homologous genomes) are grouped together into individuals
  (e.g. two nodes per diploid individual). For example, when we are working
  with polyploid samples it is useful to associate metadata with a specific
  individual rather than duplicate this information on the constituent nodes.
  See {ref}`sec_nodes_or_individuals` for more discussion on this point.

(sec_data_model_definitions_sample)=

sample
: The focal nodes of a tree sequence, usually thought of as those from which
  we have obtained data. The specification of these affects various
  methods: (1) {meth}`TreeSequence.variants` and
  {meth}`TreeSequence.haplotypes` will output the genotypes of the samples,
  and {attr}`Tree.roots` only return roots ancestral to at least one
  sample.
  (This can be checked with {meth}`~Node.is_sample`;
  see the {ref}`node table definitions <sec_node_table_definition>`
  for information on how the sample
  status a node is encoded in the `flags` column.)

(sec_data_model_definitions_edge)=

edge
: The topology of a tree sequence is defined by a set of **edges**. Each
  edge is a tuple `(left, right, parent, child)`, which records a
  parent-child relationship among a pair of nodes on the
  on the half-open interval of chromosome `[left, right)`.

(sec_data_model_definitions_site)=

site
: Tree sequences can define the mutational state of nodes as well as their
  topological relationships. A **site** is thought of as some position along
  the genome at which variation occurs. Each site is associated with
  a unique position and ancestral state.

(sec_data_model_definitions_mutation)=

mutation
: A mutation records the change of state at a particular site 'above'
  a particular node (more precisely, along the branch between the node
  in question and its parent). Each mutation is associated with a specific
  site (which defines the position along the genome), a node (which defines
  where it occurs within the tree at this position), and a derived state
  (which defines the mutational state inherited by all nodes in the subtree
  rooted at the focal node). In more complex situations in which we have
  back or recurrent mutations, a mutation must also specify its 'parent'
  mutation.

(sec_data_model_definitions_migration)=

migration
: An event at which a parent and child node were born in different populations.

(sec_data_model_definitions_population)=

population
: A grouping of nodes, e.g., by sampling location.

(sec_data_model_definitions_provenance)=

provenance
: An entry recording the origin and history of the data encoded in a tree sequence.

(sec_data_model_definitions_ID)=

ID
: In the set of interconnected tables that we define here, we refer
  throughout to the IDs of particular entities. The ID of an
  entity (e.g., a node) is defined by the position of the corresponding
  row in the table. These positions are zero indexed. For example, if we
  refer to node with ID zero, this corresponds to the node defined by the
  first row in the node table.

(sec_data_model_definitions_sequence_length)=

sequence length
: This value defines the coordinate space in which the edges and site positions
  are defined. This is most often assumed to be equal to the largest
  `right` coordinate in the edge table, but there are situations in which
  we might wish to specify the sequence length explicitly.

## Further discussion

(sec_nodes_or_individuals)=

### Nodes, Genomes, or Individuals?

The natural unit of biological analysis is (usually) the *individual*. However,
many organisms we study are diploid, and so each individual contains *two*
homologous copies of the entire genome, separately inherited from the two
parental individuals. Since each monoploid copy of the genome is inherited separately,
each diploid individual lies at the end of two distinct lineages, and so will
be represented by *two* places in any given genealogical tree. This makes it
difficult to precisely discuss tree sequences for diploids, as we have no
simple way to refer to the bundle of chromosomes that make up the "copy of the
genome inherited from one particular parent". For this reason, in this
documentation we use the non-descriptive term "node" to refer to this concept
-- and so, a diploid individual is composed of two nodes -- although we use the
term "genome" at times, for concreteness.

Several properties naturally associated with individuals are in fact assigned
to nodes in what follows: birth time and population. This is for two reasons:
First, since coalescent simulations naturally lack a notion of polyploidy, earlier
versions of `tskit` lacked the notion of an individual. Second, ancestral
nodes are not naturally grouped together into individuals -- we know they must have
existed, but have no way of inferring this grouping, so in fact many nodes in
an empirically-derived tree sequence will not be associated with individuals,
even though their birth times might be inferred.

