# Simple Makefile for building examples.
# This will build the examples in the current directory by compiling in the 
# full tskit source into each of the examples. This is *not* recommended for
# real projects!
#
# To use, type "make" in the this directory. If you have GSL installed you
# should then get two example programs built.
#
# **Note**: This repo uses git submodules, and these must be checked out
# correctly for this makefile to work, e.g.:
# 
# $ git clone git@github.com:tskit-dev/tskit.git --recurse-submodules
# 
# See the documentation (https://tskit.readthedocs.io/en/stable/c-api.html)
# for more details on how to use the C API, and the tskit build examples 
# repo (https://github.com/tskit-dev/tskit-build-examples) for examples 
# of how to set up a production-ready build with tskit.
#

CFLAGS=-I../ -I../subprojects/kastore
TSKIT_SOURCE=../tskit/*.c ../subprojects/kastore/kastore.c

all: tree_iteration haploid_wright_fisher tree_traversal

tree_iteration: tree_iteration.c 
	${CC} ${CFLAGS} -o $@ $< ${TSKIT_SOURCE} -lm

tree_traversal: tree_traversal.c 
	${CC} ${CFLAGS} -o $@ $< ${TSKIT_SOURCE} -lm

# This needs GSL
haploid_wright_fisher: haploid_wright_fisher.c 
	${CC} ${CFLAGS} -o $@ $< ${TSKIT_SOURCE} -lgsl -lgslcblas -lm

clean:
	rm -f haploid_wright_fisher tree_iteration
