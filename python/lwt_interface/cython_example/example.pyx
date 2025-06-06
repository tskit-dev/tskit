from libc.stdint cimport uint32_t
import _lwtc
import tskit

cdef extern from "tskit.h" nogil:
    ctypedef uint32_t tsk_flags_t
    ctypedef struct tsk_table_collection_t:
        pass
    ctypedef struct tsk_treeseq_t:
        pass
    int tsk_treeseq_init(tsk_treeseq_t *self, const tsk_table_collection_t *tables, tsk_flags_t options)
    int tsk_treeseq_free(tsk_treeseq_t *self)
    int tsk_table_collection_build_index(tsk_table_collection_t *self, tsk_flags_t options)
    ctypedef struct tsk_tree_t:
        pass
    int tsk_tree_init(tsk_tree_t *self, const tsk_treeseq_t *ts, tsk_flags_t options)
    int tsk_tree_first(tsk_tree_t *self)
    int tsk_tree_next(tsk_tree_t *self)
    int tsk_tree_last(tsk_tree_t *self)
    int tsk_tree_prev(tsk_tree_t *self)
    int tsk_tree_get_num_roots(tsk_tree_t *self)
    int tsk_tree_free(tsk_tree_t *self)
    const char *tsk_strerror(int err)

cdef extern:
    ctypedef class _lwtc.LightweightTableCollection [object LightweightTableCollection]:
        cdef tsk_table_collection_t *tables

def check_tsk_error(val):
    if val < 0:
        raise RuntimeError(tsk_strerror(val))

def iterate_trees(pyts: tskit.TreeSequence):
    lwtc = LightweightTableCollection()
    lwtc.fromdict(pyts.dump_tables().asdict())        
    cdef tsk_treeseq_t ts
    err = tsk_treeseq_init(&ts, lwtc.tables, 0)
    check_tsk_error(err)
    cdef tsk_tree_t tree
    ret = tsk_tree_init(&tree, &ts, 0)
    check_tsk_error(ret)

    print("Iterate forwards")
    cdef int tree_iter = tsk_tree_first(&tree)
    while tree_iter == 1:
        print("\ttree has %d roots" % (tsk_tree_get_num_roots(&tree)))
        tree_iter = tsk_tree_next(&tree)
    check_tsk_error(tree_iter)

    print("Iterate backwards")
    tree_iter = tsk_tree_last(&tree)
    while tree_iter == 1:
        print("\ttree has %d roots" % (tsk_tree_get_num_roots(&tree)))
        tree_iter = tsk_tree_prev(&tree)
    check_tsk_error(tree_iter)

    tsk_tree_free(&tree)
    tsk_treeseq_free(&ts)

def main():
    import msprime as msp  # (msprime could be compiled against a different version of tskit)
    ts = msp.simulate(sample_size=5, length=100, recombination_rate=.01)  
    iterate_trees(ts)
