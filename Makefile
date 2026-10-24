.PHONY: setup test inspect
setup:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
test:
	pytest -q
inspect:
	python -m creditrisk.inspect_files --data-dir data/raw
