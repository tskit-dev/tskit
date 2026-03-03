
all: cmodule

allchecks: example_c_module.c
	CFLAGS="-std=c99 --coverage -Wall -Wextra -Werror -Wno-unused-parameter -Wno-cast-function-type" \
	uv run --project=../ python setup.py build_ext --inplace

cmodule: example_c_module.c
	uv run --project=../ python setup.py build_ext --inplace

clean:
	rm -f *.so *.o tags
	rm -fR build
