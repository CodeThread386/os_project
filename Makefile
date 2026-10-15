# Common tasks. `make help` lists them.
VENV ?= .venv
PY   := $(VENV)/bin/python
AL   := $(VENV)/bin/agentlayer

.PHONY: help setup test lint doctor demo ui eval top clean

help:
	@echo "setup   create the virtualenv and install AgentLayer"
	@echo "test    run the test suite (no network needed)"
	@echo "lint    pyflakes over the code and tests"
	@echo "doctor  check the environment"
	@echo "demo    the five phase policies side by side"
	@echo "ui      open the web dashboard"
	@echo "eval    live evaluation: tasks and attacks (uses the model)"
	@echo "top     terminal system monitor"

setup:
	./setup.sh

test:
	$(PY) -m pytest

lint:
	$(PY) -m pyflakes agentlayer tests

doctor:
	$(AL) doctor

demo:
	$(AL) demo

ui:
	$(AL) ui

eval:
	$(AL) eval

top:
	$(AL) top

clean:
	rm -rf build dist *.egg-info .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
