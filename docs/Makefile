BUILDDIR    = _build
DOXYGEN_XML = doxygen/xml

all: ${DOXYGEN_XML}
	./build.sh

${DOXYGEN_XML}: ../c/tskit/*.h
	cd doxygen && doxygen

clean:
	rm -fR $(BUILDDIR) $(DOXYGEN_XML)
