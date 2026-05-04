.PHONY: setup lint format test clean preprocess train generate

# Detect a TF-compatible Python (3.10 or 3.11). Override with: make setup PYTHON=/path/to/python
PYTHON ?= $(shell \
  python3.10 --version >/dev/null 2>&1 && echo python3.10 || \
  (python3.11 --version >/dev/null 2>&1 && echo python3.11) || \
  (python3 --version >/dev/null 2>&1 && echo python3))

setup:
	$(PYTHON) -m venv venv
	. venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt
	mkdir -p data/raw data/processed/sequences outputs/models outputs/midi outputs/logs

lint:
	flake8 src/ tests/ --max-line-length 120
	isort --check-only src/ tests/

format:
	black src/ tests/ --line-length 120
	isort src/ tests/

test:
	pytest tests/ -v || test $$? -eq 5

preprocess:
	. venv/bin/activate && python -m src.preprocessing

train:
	. venv/bin/activate && python -m src.train

generate:
	. venv/bin/activate && python -m src.generate \
		--model outputs/models/best_model.keras \
		--output outputs/midi/generated.midi \
		--vocab data/processed/vocabulary.json \
		--length 512 \
		--temperature 1.0

clean:
	rm -rf __pycache__ .pytest_cache outputs/logs/*
	find . -name "*.pyc" -delete
