# LightweightTableCollection interface

The files in this directory define the LightweightTableCollection
interface used to safely interchange table collection data between 
different compiled instances of the tskit C library. This is a 
*very* specialised use-case, and unless you are using the tskit
C API in your own compiled Python module (either via Cython
or the Python C API), you almost certainly don't need to use
this code.

## TODO document
