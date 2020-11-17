# LightweightTableCollection interface

The files in this directory define the LightweightTableCollection
interface used to safely interchange table collection data between 
different compiled instances of the tskit C library. This is a 
*very* specialised use-case, and unless you are using the tskit
C API in your own compiled Python module (either via Cython
or the Python C API), you almost certainly don't need to use
this code.

## Overview

To allow a tskit table collection to be transferred from one compiled Python
extension module to another the table collection is converted to a `dict` of
basic python types and numpy arrays. This is then converted back in the receiving
module. `tskit_lwt_interface.h` provides a function `register_lwt_class` that 
defines a Python class `LightweightTableCollection` that performs these conversions
with methods `asdict` and `fromdict`. These methods mirror the `asdict` and `fromdict`
methods on `tskit.TableCollection`.

## Usage
An example C module skeleton `example_c_module.c` is provided, which shows passing tables
to the C module. See `test_example_c_module.py` for the python example usage
of the example module.

To add the 
`LightweightTableCollection` type to your module you include `tskit_lwt_interface.h`
and then call `register_lwt_class` on your C Python module object. You can then convert
to and from the lightweight table collection in Python, for example to convert a tskit
`TableCollection` to a `LightweightTableCollection`:
```python
tables = tskit.TableCollection(1)
lwt = example_c_module.LightweightTableCollection()
lwt.fromdict(tables.asdict())
```
and vice-versa:
```python
tc = tskit.TableCollection(lwt.asdict())
```
In C you can access the tables in a `LightweightTableCollection` instance that is passed 
to your function, as shown in the `example_receiving` function in `example_c_module.c`. 
Note the requirement to check for errors from tskit functions and to call
`handle_tskit_error` to set a Python error, returning `NULL` to Python to indicate error.

Tables can also be modified in the extension code as in `example_modifying`. We recommend
creating table collections in Python then passing them to C for modification rather than
creating them in C and returning them. This avoids complex managing of object lifecycles
in C code.




    



