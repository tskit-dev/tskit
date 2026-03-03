all: ext3

allchecks: _tskitmodule.c
	CFLAGS="-std=c99 --coverage -Wall -Wextra -Werror -Wno-unused-parameter -Wno-cast-function-type" \
	uv run python setup.py build_ext --inplace

ext3: _tskitmodule.c
	uv run python setup.py build_ext --inplace

ctags:
	ctags lib/*.c lib/*.h tskit/*.py

clean:
	rm -f *.so *.o tags
	rm -fR build
