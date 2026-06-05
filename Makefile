.PHONY: smoke

PYTHON ?= python3

smoke:
	$(PYTHON) -m compileall src tests benchmarks
