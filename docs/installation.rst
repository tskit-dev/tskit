.. _sec_installation:

============
Installation
============

.. note:: This documentation is incomplete. Once we have a conda package that
    is installable we'll update the documentation to use that route also. However,
    as there are no external dependencies, pip should work well for all
    non-Windows users.

The tree sequence toolkit is often installed to provide succinct tree sequence functionality
to other software (such as `msprime <https://github.com/tskit-dev/msprime>`_). If installing
as a standalone Python module, users are encouraged to install an official release from PyPI
using pip::

    $ python -m pip install tskit

For technical reasons it is not possible to install a development version directly using the
GitHub URL (i.e. ``pip install git+git://...`` will not work)
