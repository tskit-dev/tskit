
all: ext3

ext3: _tskitmodule.c 
	# CFLAGS="-std=c99 -Wall -Wextra -Werror -Wno-unused-parameter" \
	#        python3 setup.py build_ext --inplace
	# Disable checks for now.
	python3 setup.py build_ext --inplace

ext2: _tskitmodule.c 
	python2 setup.py build_ext --inplace

ctags:
	ctags lib/*.c lib/*.h tskit/*.py

clean:
	rm -f *.so *.o tags
	rm -fR build
