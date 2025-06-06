all: compile run

compile:
	pip install -e . --use-pep517

run:
	PYTHONPATH=../.. python -c "import example; example.main()"

clean:
	rm -rf build/
	rm -rf tskit_cython_example.egg-info/
	rm -f example.c
	rm -f *.so
