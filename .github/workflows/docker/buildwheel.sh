#!/bin/bash
DOCKER_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$DOCKER_DIR/shared.env"

set -e -x

ARCH=`uname -p`
echo "arch=$ARCH"
#yum -y install gsl-devel #For msprime

cd python

for V in "${PYTHON_VERSIONS[@]}"; do
    PYBIN=/opt/python/$V/bin
    rm -rf build/       # Avoid lib build by narrow Python is used by wide python
    # Instead of letting setup.py install a newer numpy we install it here
    # using the oldest supported version for ABI compatibility
    $PYBIN/pip install oldest-supported-numpy
    $PYBIN/python setup.py build_ext --inplace
    $PYBIN/python setup.py bdist_wheel
done

cd dist
for whl in *.whl; do
    auditwheel repair "$whl"
    rm "$whl"
done