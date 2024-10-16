PYTHON := $(shell command -v python 2> /dev/null)
ifndef PYTHON
$(error "python is not available via the `python` executable. If you have python only via `python3` you may need to `apt install python-is-python3`")
endif

all: ext3

allchecks: _tskitmodule.c
	CFLAGS="-std=c99 --coverage -Wall -Wextra -Werror -Wno-unused-parameter -Wno-cast-function-type" \
	python setup.py build_ext --inplace

ext3: _tskitmodule.c
	
	python setup.py build_ext --inplace

ctags:
	ctags lib/*.c lib/*.h tskit/*.py

clean:
	rm -f *.so *.o tags
	rm -fR build
