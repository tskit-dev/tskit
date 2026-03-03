all: compile run

compile:
	uv run --project=../../ --group=test-lwt setup.py build_ext --inplace

run:
	uv run --project=../../ --group=test-lwt python -c "import example; example.main()"

clean:
	rm -rf build/
	rm -rf tskit_cython_example.egg-info/
	rm -f example.c
	rm -f *.so
