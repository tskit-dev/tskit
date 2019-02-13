.. _sec_tutorial:

========
Tutorial
========

.. todo:: The content here has been ported from the msprime tutorial and
    needs to be reorganised to make a coherent narrative.


.. _sec_tutorial_traversing_trees:

****************
Traversing trees
****************

A :class:`.Tree` represents a single tree in a :class:`.TreeSequence`.
The ``tskit`` Tree implementation differs from most tree libraries by
using **integer IDs** to refer to nodes rather than objects. Thus, when we wish to
find the parent of the node with ID '0', we use ``tree.parent(0)``, which
returns another integer. If '0' does not have a parent in the current tree
(e.g., if it is a root), then the special value :const:`.NULL`
(:math:`-1`) is returned. The children of a node are found using the
:meth:`.Tree.children` method. To obtain information about a particular node,
one may either use ``tree.tree_sequence.node(u)`` to which returns the
corresponding :class:`Node` instance, or use the :meth:`.Tree.time` or
:meth:`.Tree.population` shorthands. Tree traversals in various orders
is possible using the :meth:`.Tree.nodes` iterator.

.. todo:: Add tree diagram and example. Also describe the left_child,
    right_child, left_sib, right_sib functions.


.. _sec_tutorial_moving_along_a_tree_sequence:

****************************
Moving along a tree sequence
****************************

Most of the time we will want to iterate over all the trees in a tree sequence
sequentially as efficiently as possible. The simplest way to do this is to
use the :meth:`.TreeSequence.trees` method:

.. code-block:: python

    import msprime

    ts = msprime.simulate(5, recombination_rate=1, random_seed=42)

    print("Tree sequence has {} trees".format(ts.num_trees))
    print()
    for tree in ts.trees():
        print("Tree {} covers [{:.2f}, {:.2f}); TMRCA = {:.4f}".format(
            tree.index, *tree.interval, tree.time(tree.root)))

Running the code, we get::

    Tree sequence has 7 trees

    Tree 0 covers [0.00, 0.08); TMRCA = 4.2542
    Tree 1 covers [0.08, 0.27); TMRCA = 2.5973
    Tree 2 covers [0.27, 0.37); TMRCA = 4.2542
    Tree 3 covers [0.37, 0.66); TMRCA = 2.5973
    Tree 4 covers [0.66, 0.71); TMRCA = 4.2542
    Tree 5 covers [0.71, 0.75); TMRCA = 2.5973
    Tree 6 covers [0.75, 1.00); TMRCA = 2.5973

Here we run a small simulation using `msprime <https://msprime.readthedocs.io>`_
which results in 7 distinct trees along a genome of length 1. We then iterate
over these trees sequentially using the :meth:`.TreeSequence.trees` method,
and print out each tree's index, the interval over which the tree applies
and the time of the most recent common ancestor of all the samples. This
method is very efficient, and allows us to quickly iterate over very large
tree sequences.

We can also efficiently iterate over the trees backwards, using Python's
:func:`reversed` function:

.. code-block:: python

    for tree in reversed(ts.trees()):
        print("Tree {} covers [{:.2f}, {:.2f}); TMRCA = {:.4f}".format(
            tree.index, *tree.interval, tree.time(tree.root)))

giving::

    Tree 6 covers [0.75, 1.00); TMRCA = 2.5973
    Tree 5 covers [0.71, 0.75); TMRCA = 2.5973
    Tree 4 covers [0.66, 0.71); TMRCA = 4.2542
    Tree 3 covers [0.37, 0.66); TMRCA = 2.5973
    Tree 2 covers [0.27, 0.37); TMRCA = 4.2542
    Tree 1 covers [0.08, 0.27); TMRCA = 2.5973
    Tree 0 covers [0.00, 0.08); TMRCA = 4.2542

One of the reasons that the ``trees`` iterator allows us to access
the trees in a tree sequence so efficiently is because we use the
same underlying instance of the ``.Tree`` class each time. That is,
each time the iterator returns a value, it is actually the same tree
instance each time which has been updated internally to reflect the
(usually small) changes in the tree along the sequence. As a
result of this, if we store the results of the tree iterator in a
list, we will get unexpected results:

.. code-block:: python

    for tree in list(ts.trees()):
        print("Tree {} covers [{:.2f}, {:.2f}): id={:x}".format(
            tree.index, *tree.interval, id(tree)))

::

    Tree -1 covers [0.00, 0.00): id=7f290becb3c8
    Tree -1 covers [0.00, 0.00): id=7f290becb3c8
    Tree -1 covers [0.00, 0.00): id=7f290becb3c8
    Tree -1 covers [0.00, 0.00): id=7f290becb3c8
    Tree -1 covers [0.00, 0.00): id=7f290becb3c8
    Tree -1 covers [0.00, 0.00): id=7f290becb3c8
    Tree -1 covers [0.00, 0.00): id=7f290becb3c8

We have stored seven copies of the same :class:`Tree` instance in the
list. Because iteration has ended, this tree is in the "null" state (see
below for more details) which means that it doesn't represent any of the
trees in the tree sequence.

If we do wish to obtain a list of the trees, we can do so by using the
:meth:`.TreeSequence.aslist` method:

.. code-block:: python

    for tree in ts.aslist():
        print("Tree {} covers [{:.2f}, {:.2f}): id={:x}".format(
            tree.index, *tree.interval, id(tree)))

::

    Tree 0 covers [0.00, 0.08): id=7fd2c50a40f0
    Tree 1 covers [0.08, 0.27): id=7fd2b2aca6d8
    Tree 2 covers [0.27, 0.37): id=7fd2b2adde10
    Tree 3 covers [0.37, 0.66): id=7fd2b2adddd8
    Tree 4 covers [0.66, 0.71): id=7fd2b2addd68
    Tree 5 covers [0.71, 0.75): id=7fd2b2addcf8
    Tree 6 covers [0.75, 1.00): id=7fd2b2addeb8

Note that we now have a different object for each tree in the list. Please
note that this is **much** less efficient than iterating over the trees
using the :meth:`.TreeSequence.trees` method (and uses far more memory!),
and should only be used as a convenience when working with small trees.

We can also obtain specific trees along the sequence, using the
:meth:`.TreeSequence.first`,
:meth:`.TreeSequence.last`
:meth:`.TreeSequence.at` and
:meth:`.TreeSequence.at_index` methods. The ``first()`` and ``last()``
methods return the first and last trees in the sequence, as might be
imagined. The ``at()`` method returns the tree that covers a
given genomic location, and the ``at_index()`` method returns the
tree at a given index along the sequence:

.. code-block:: python

    tree = ts.at(0.5)
    print("Tree {} covers [{:.2f}, {:.2f}): id={:x}".format(
        tree.index, *tree.interval, id(tree)))
    tree = ts.at_index(0)
    print("Tree {} covers [{:.2f}, {:.2f}): id={:x}".format(
        tree.index, *tree.interval, id(tree)))
    tree = ts.at_index(-1)
    print("Tree {} covers [{:.2f}, {:.2f}): id={:x}".format(
        tree.index, *tree.interval, id(tree)))

::

    Tree 3 covers [0.37, 0.66): id=7f9fdb469630
    Tree 0 covers [0.00, 0.08): id=7f9fdb46d160
    Tree 6 covers [0.75, 1.00): id=7f9fdb469630

Note that each call to these methods return a different :class:`.Tree` instance
and so it is much, much less efficient to sequentially access trees
by their index values than it is to use the :meth:`.TreeSequence.trees`
iterator.


**********************
Editing tree sequences
**********************

Sometimes we wish to make some minor modifications to a tree sequence that has
been generated by a simulation. However, tree sequence objects are **immutable**
and so we cannot edit a them in place. To modify a tree sequence, we need to
extract the underlying :ref:`tables <sec_table_definitions>` of information, edit these tables,
and then create a new tree sequence from them.
These tables succinctly store everything we need to know
about a tree sequence, and can be manipulated using the :ref:`sec_tables_api`.
In the following example, we use this approach
to remove all singleton sites from a given tree sequence.

.. code-block:: python

    def strip_singletons(ts):
        tables = ts.dump_tables()
        tables.sites.clear()
        tables.mutations.clear()
        for tree in ts.trees():
            for site in tree.sites():
                assert len(site.mutations) == 1  # Only supports infinite sites muts.
                mut = site.mutations[0]
                if tree.num_samples(mut.node) > 1:
                    site_id = tables.sites.add_row(
                        position=site.position,
                        ancestral_state=site.ancestral_state)
                    tables.mutations.add_row(
                        site=site_id, node=mut.node, derived_state=mut.derived_state)
        return tables.tree_sequence()


This function takes a tree sequence containing some infinite sites mutations as
input, and returns a copy in which all singleton sites have been removed.
The approach is very simple: we get a copy of the underlying
table data in a :class:`.TableCollection` object, and first clear the
site and mutation tables. We then consider each site in turn,
and if the allele frequency of
the mutation is greater than one, we add the site and mutation to our
output tables using :meth:`.SiteTable.add_row` and :meth:`.MutationTable.add_row`.
(In this case we consider only simple infinite sites mutations,
where we cannot have back or recurrent mutations. These would require a slightly
more involved approach where we keep a map of mutation IDs so that
mutation ``parent`` values could be computed. We have also omitted the
site and mutation metadata in the interest of simplicity.)

After considering each site, we then create a new tree sequence using
the :meth:`.TableCollection.tree_sequence` method on our updated tables.
Using this function then, we get::

    >>> ts = msprime.simulate(10, mutation_rate=10)
    >>> ts.num_sites
    50
    >>> ts_new = strip_singletons(ts)
    >>> ts_new.num_sites
    44
    >>>

Thus, we have removed 6 singleton sites from the tree sequence.

.. todo::

    Add another example here where we use the array oriented API to edit
    the nodes and edges of a tree sequence. Perhaps decapitating would be a
    good example?

*******************
Working with Tables
*******************


Tables provide a convenient method for viewing, importing and exporting tree
sequences, and are closely tied to the underlying data structures.
There are eight tables that together define a tree sequence,
although some may be empty,
and together they form a :class:`TableCollection`.
The tables are defined in :ref:`Table Definitions <sec_table_definitions>`,
and the :ref:`Tables API <sec_tables_api>` section describes how to work with them.
Here we make some general remarks about what you can, and cannot do with them.


``tskit`` provides direct access to the columns of each table as
``numpy`` arrays: for instance, if ``n`` is a ``NodeTable``, then ``n.time``
will return an array containing the birth times of the individuals whose genomes
are represented by the nodes in the table.
*However*, it is important to note that this is *not* a shallow copy:
modifying ``n.time`` will *not* change the node table ``n``.  This may change in
the future, but currently there are three ways to modify tables: ``.add_row()``,
``.set_columns()``, and ``.append_columns()``
(and also ``.clear()``, which empties the table).

For example, a node table could be constructed using ``.add_row()`` as
follows::

    n = tskit.NodeTable()
    sv = [True, True, True, False, False, False, False]
    tv = [0.0, 0.0, 0.0, 0.4, 0.5, 0.7, 1.0]
    pv = [0, 0, 0, 0, 0, 0, 0]
    for s, t, p in zip(sv, tv, pv):
        n.add_row(flags=s, population=p, time=t)


obtaining::

    >>> print(n)
    id    flags    population    individual    time    metadata
    0    1    0    -1    0.0
    1    1    0    -1    0.0
    2    1    0    -1    0.0
    3    0    0    -1    0.4
    4    0    0    -1    0.5
    5    0    0    -1    0.7
    6    0    0    -1    1.0


The ``.add_row()`` method is natural (and should be reasonably efficient) if
new records appear one-by-one. In the example above it would have been more
natural to use ``.set_columns()`` - equivalently::

    n = tskit.NodeTable()
    n.set_columns(flags=sv, population=pv, time=tv)

Since columns cannot be modified directly as properties of the tables,
they must be extracted, modified, then replaced.
For example, here we add 1.4 to every ``time`` except the first
in the node table constructed above (using ``numpy`` indexing)::

    tn = n.time
    tn[1:] = tn[1:] + 1.4
    n.set_columns(flags=n.flags, population=n.population, time=tn)

The result is::

    >>> print(n)
    id    flags    population    individual    time    metadata
    0    1    0    -1    0.0
    1    1    0    -1    1.4
    2    1    0    -1    1.4
    3    0    0    -1    1.8
    4    0    0    -1    1.9
    5    0    0    -1    2.1
    6    0    0    -1    2.4


*****************************
Overview of the Tables Format
*****************************

The :ref:`Table Definitions <sec_table_definitions>` section gives a precise
definition of how a tree sequence is stored in a collection of tables.
Here we give an overview. Consider the following sequence of trees::

    time ago
    --------
       1.0         6
                 ┏━┻━━┓
                 ┃    ┃
       0.7       ┃    ╋                     5
                 ┃    ┃                   ┏━┻━┓
       0.5       ┃    4         4         ┃   4
                 ┃  ┏━┻━┓     ┏━┻━┓       ┃ ┏━┻━┓
                 ┃  ┃   ┃     ┃   ╋       ┃ ┃   ┃
       0.4       ┃  ┃   ┃     ┃   3       ┃ ┃   ┃
                 ┃  ┃   ┃     ┃ ┏━┻━┓     ┃ ┃   ┃
                 ┃  ┃   ┃     ┃ ┃   ╋     ┃ ┃   ┃
       0.0       0  1   2     1 0   2     0 1   2

    position 0.0          0.2         0.8         1.0

Ancestral recombination events have produced three different trees
that relate the three sampled genomes ``0``, ``1``, and ``2`` to each other
along the chromosome of length 1.0.

Each node in each of the above trees represents a particular ancestral genome
(a *haploid* genome; diploid individuals would be represented by two nodes).
We record when each of nodes lived in a :class:`NodeTable`::

    NodeTable:

    id      flags    population   time
    0       1        0            0
    1       1        0            0
    2       1        0            0
    3       0        0            0.4
    4       0        0            0.5
    5       0        0            0.7
    6       0        0            1.0

Importantly, the first column, ``id``, is not actually recorded, and is
only shown when printing out node tables (as here) for convenience.
The second column, ``flags`` records a ``1`` for the individuals that are *samples*,
i.e., whose entire genealogical history is recorded by these trees.
(Note that the trees above record that node 3 inherited from node 4
on the middle portion of the genome, but not on the ends.)

We next need to record each tree's edges. Since some edges are present
in more than one tree (e.g., node 1 inherits from node 4 across
the entire sequence), we record in the :class:`EdgeTable` each edge
and the genomic region for which it appears in the trees::


    EdgeTable:

    left    right   parent  children
    0.2     0.8     3       0
    0.2     0.8     3       2
    0.0     1.0     4       1
    0.0     0.2     4       2
    0.8     1.0     4       2
    0.2     0.8     4       3
    0.8     1.0     5       0
    0.8     1.0     5       4
    0.0     0.2     6       0
    0.0     0.2     6       4

Since node 3 is most recent, the edge that says that nodes 0 and 2 inherit
from node 3 on the interval between 0.2 and 0.8 comes first.  Next are the
edges from node 4: there are six of these, two for each of the three genomic
intervals over which node 4 is ancestor to a distinct set of nodes.  At this
point, we know the full tree on the middle interval.  Finally, edges
specifying the common ancestor of 0 and 4 on the remaining intervals (parents 6
and 5 respectively) allow us to construct all trees across the entire interval.

There are three mutations in the depiction above,
marked by ``╋``: one above node ``4`` on the first tree,
and the other two above nodes ``2`` and ``3`` on the second tree.
Suppose that the first mutation occurs at position 0.1 and the mutations in the
second tree both occurred at the same position, at 0.5 (with a back mutation).
To record the inheritance patterns of these, we need only record
the positions on the genome at which they occurred,
and on which edge (equivalently, above which node) they occurred.
The positions are recorded in the :class:`SiteTable`::

    SiteTable:

    id    position    ancestral_state
    0    0.1         0
    1    0.5         0

As with node tables, the ``id`` column is **not** actually recorded, but is
implied by the position in the table.  The results of the
actual mutations are then recorded::

    MutationTable:

    site    node    derived_state
    0        4        1
    1        3        1
    1        2        0

This would then result in the following (two-locus) haplotypes for the three
samples::

    sample  haplotype
    ------  ---------
    0       01
    1       10
    2       10


To create these tables, and the corresponding tree sequence, we would
create a :class:`TableCollection`, and then use its
:meth:`TableCollection.tree_sequence` method::

    tables = tskit.TableCollection(sequence_length=1.0)

    # Nodes
    sv = [True, True, True, False, False, False, False]
    tv = [0.0, 0.0, 0.0, 0.4, 0.5, 0.7, 1.0]

    for is_sample, t in zip(sv, tv):
     flags = tskit.NODE_IS_SAMPLE if is_sample else 0
     tables.nodes.add_row(flags=flags, time=t)

    # Edges
    lv = [0.2, 0.2, 0.0, 0.2, 0.8, 0.0, 0.8, 0.2, 0.8, 0.8, 0.0, 0.0]
    rv = [0.8, 0.8, 0.2, 0.8, 1.0, 0.2, 1.0, 0.8, 1.0, 1.0, 0.2, 0.2]
    pv = [3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 6, 6]
    cv = [0, 2, 1, 1, 1, 2, 2, 3, 0, 4, 0, 4]

    for l, r, p, c in zip(lv, rv, pv, cv):
        tables.edges.add_row(left=l, right=r, parent=p, child=c)

    # Sites
    for p, a in zip([0.1, 0.5], ['0', '0']):
        tables.sites.add_row(position=p, ancestral_state=a)

    # Mutations
    for s, n, d in zip([0, 1, 1], [4, 3, 2], ['1', '1', '0']):
        tables.mutations.add_row(site=s, node=n, derived_state=d)

We can then finally obtain the tree sequence::

    ts = tables.tree_sequence()
    for t in ts.trees():
      print(t.draw(format='unicode'))


**************
Calculating LD
**************

The ``tskit`` API provides methods to efficiently calculate
population genetics statistics. For example, the :class:`.LdCalculator`
class allows us to compute pairwise `linkage disequilibrium
<https://en.wikipedia.org/wiki/Linkage_disequilibrium>`_ coefficients.
Here we use the :meth:`.LdCalculator.r2_matrix` method to easily make an
LD plot using `matplotlib <http://matplotlib.org/>`_. (Thanks to
the excellent `scikit-allel
<http://scikit-allel.readthedocs.io/en/latest/index.html>`_
for the basic `plotting code
<http://scikit-allel.readthedocs.io/en/latest/_modules/allel/stats/ld.html#plot_pairwise_ld>`_
used here.)

.. code-block:: python

    import msprime
    import tskit
    import matplotlib.pyplot as pyplot

    def ld_matrix_example():
        ts = msprime.simulate(100, recombination_rate=10, mutation_rate=20,
                random_seed=1)
        ld_calc = tskit.LdCalculator(ts)
        A = ld_calc.r2_matrix()
        # Now plot this matrix.
        x = A.shape[0] / pyplot.rcParams['figure.dpi']
        x = max(x, pyplot.rcParams['figure.figsize'][0])
        fig, ax = pyplot.subplots(figsize=(x, x))
        fig.tight_layout(pad=0)
        im = ax.imshow(A, interpolation="none", vmin=0, vmax=1, cmap="Blues")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in 'top', 'bottom', 'left', 'right':
            ax.spines[s].set_visible(False)
        pyplot.gcf().colorbar(im, shrink=.5, pad=0)
        pyplot.savefig("ld.svg")


.. image:: _static/ld.svg
   :width: 800px
   :alt: An example LD matrix plot.

.. _sec_tutorial_threads:

********************
Working with threads
********************

When performing large calculations it's often useful to split the
work over multiple processes or threads. The ``tskit`` API can
be used without issues across multiple processes, and the Python
:mod:`multiprocessing` module often provides a very effective way to
work with many replicate simulations in parallel.

When we wish to work with a single very large dataset, however, threads can
offer better resource usage because of the shared memory space. The Python
:mod:`threading` library gives a very simple interface to lightweight CPU
threads and allows us to perform several CPU intensive tasks in parallel. The
``tskit`` API is designed to allow multiple threads to work in parallel when
CPU intensive tasks are being undertaken.

.. note:: In the CPython implementation the `Global Interpreter Lock
   <https://wiki.python.org/moin/GlobalInterpreterLock>`_ ensures that
   only one thread executes Python bytecode at one time. This means that
   Python code does not parallelise well across threads, but avoids a large
   number of nasty pitfalls associated with multiple threads updating
   data structures in parallel. Native C extensions like ``numpy`` and ``tskit``
   release the GIL while expensive tasks are being performed, therefore
   allowing these calculations to proceed in parallel.

In the following example we wish to find all mutations that are in approximate
LD (:math:`r^2 \geq 0.5`) with a given set of mutations. We parallelise this
by splitting the input array between a number of threads, and use the
:meth:`.LdCalculator.r2_array` method to compute the :math:`r^2` value
both up and downstream of each focal mutation, filter out those that
exceed our threshold, and store the results in a dictionary. We also
use the very cool `tqdm <https://pypi.python.org/pypi/tqdm>`_ module to give us a
progress bar on this computation.

.. code-block:: python

    import threading
    import numpy as np
    import tqdm
    import msprime
    import tskit

    def find_ld_sites(
            tree_sequence, focal_mutations, max_distance=1e6, r2_threshold=0.5,
            num_threads=8):
        results = {}
        progress_bar = tqdm.tqdm(total=len(focal_mutations))
        num_threads = min(num_threads, len(focal_mutations))

        def thread_worker(thread_index):
            ld_calc = tskit.LdCalculator(tree_sequence)
            chunk_size = int(math.ceil(len(focal_mutations) / num_threads))
            start = thread_index * chunk_size
            for focal_mutation in focal_mutations[start: start + chunk_size]:
                a = ld_calc.r2_array(
                    focal_mutation, max_distance=max_distance,
                    direction=tskit.REVERSE)
                rev_indexes = focal_mutation - np.nonzero(a >= r2_threshold)[0] - 1
                a = ld_calc.r2_array(
                    focal_mutation, max_distance=max_distance,
                    direction=tskit.FORWARD)
                fwd_indexes = focal_mutation + np.nonzero(a >= r2_threshold)[0] + 1
                indexes = np.concatenate((rev_indexes[::-1], fwd_indexes))
                results[focal_mutation] = indexes
                progress_bar.update()

        threads = [
            threading.Thread(target=thread_worker, args=(j,))
            for j in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        progress_bar.close()
        return results

    def threads_example():
        ts = msprime.simulate(
            sample_size=1000, Ne=1e4, length=1e7, recombination_rate=2e-8,
            mutation_rate=2e-8)
        counts = np.zeros(ts.num_sites)
        for tree in ts.trees():
            for site in tree.sites():
                assert len(site.mutations) == 1
                mutation = site.mutations[0]
                counts[site.id] = tree.num_samples(mutation.node)
        doubletons = np.nonzero(counts == 2)[0]
        results = find_ld_sites(ts, doubletons, num_threads=8)
        print(
            "Found LD sites for", len(results), "doubleton sites out of",
            ts.num_sites)

In this example, we first simulate 1000 samples of 10 megabases and find all
doubleton mutations in the resulting tree sequence. We then call the
``find_ld_sites()`` function to find all mutations that are within 1 megabase
of these doubletons and have an :math:`r^2` statistic of greater than 0.5.

The ``find_ld_sites()`` function performs these calculations in parallel using
8 threads. The real work is done in the nested ``thread_worker()`` function,
which is called once by each thread. In the thread worker, we first allocate an
instance of the :class:`.LdCalculator` class. (It is **critically important**
that each thread has its own instance of :class:`.LdCalculator`, as the threads
will not work efficiently otherwise.) After this, each thread works out the
slice of the input array that it is responsible for, and then iterates over
each focal mutation in turn. After the :math:`r^2` values have been calculated,
we then find the indexes of the mutations corresponding to values greater than
0.5 using :func:`numpy.nonzero`. Finally, the thread stores the resulting array
of mutation indexes in the ``results`` dictionary, and moves on to the next
focal mutation.


Running this example we get::

    >>> threads_example()
    100%|████████████████████████████████████████████████| 4045/4045 [00:09<00:00, 440.29it/s]
    Found LD sites for 4045 doubleton mutations out of 60100
