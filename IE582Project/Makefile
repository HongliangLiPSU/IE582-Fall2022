.PHONY: venv install run run-module run-map clean

venv:
	python3 -m venv .venv

install:
	. .venv/bin/activate && python -m pip install --upgrade pip && pip install -r requirements.txt

run:
	. .venv/bin/activate && python build_food_insecurity_index.py

run-module:
	. .venv/bin/activate && PYTHONPATH=src python -m ie582_food_insecurity.cli

run-map:
	. .venv/bin/activate && PYTHONPATH=src python -m ie582_food_insecurity.map_cli

clean:
	rm -rf __pycache__ src/**/__pycache__ .pytest_cache
