
all: cmodule

allchecks: example_c_module.c
	CFLAGS="-std=c99 --coverage -Wall -Wextra -Werror -Wno-unused-parameter -Wno-cast-function-type" \
	python3 setup.py build_ext --inplace

cmodule: example_c_module.c
	python3 setup.py build_ext --inplace

clean:
	rm -f *.so *.o tags
	rm -fR build
