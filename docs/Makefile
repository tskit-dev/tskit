
# Need to set PYTHONPATH so that we pick up the local tskit
PYPATH=$(shell pwd)/../python/
TSK_VERSION:=$(shell PYTHONPATH=${PYPATH} \
   python3 -c 'import tskit; print(tskit.__version__.split("+")[0])')

BUILDDIR      = _build
DOXYGEN_XML=doxygen/xml

all: ${DOXYGEN_XML} dev

${DOXYGEN_XML}: ../c/tskit/*.h
	cd doxygen && doxygen

dev:
	PYTHONPATH=${PYPATH} ./build.sh

dist:
	@echo Building distribution for tskit version ${TSK_VERSION}
	cd doxygen && doxygen
	sed -i -e s/__TSKIT_VERSION__/${TSK_VERSION}/g _config.yml
	PYTHONPATH=${PYPATH} ./build.sh

clean:
	rm -fR $(BUILDDIR) $(DOXYGEN_XML)
