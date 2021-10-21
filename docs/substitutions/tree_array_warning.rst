.. warning:: This is a high-performance interface which
    provides zero-copy access to memory used in the C library.
    As a consequence, the values stored in this array will change as
    the Tree state is modified as we move along the tree sequence. See the
    :class:`.Tree` documentation for more details. Therefore, if you want to
    compare arrays representing different trees along the sequence, you must
    take **copies** of the arrays.
