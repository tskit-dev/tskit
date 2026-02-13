BUILDDIR      = _build
DOXYGEN_XML=doxygen/xml

all: ${DOXYGEN_XML} dev

${DOXYGEN_XML}: ../c/tskit/*.h
	cd doxygen && doxygen

dev:
	uv --project=../python run jupyter-book build -Wn --keep-going .

clean:
	rm -fR $(BUILDDIR) $(DOXYGEN_XML)
