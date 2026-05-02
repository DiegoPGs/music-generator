.PHONY: setup lint format test clean

setup:
	python -m venv venv
	. venv/bin/activate && pip install -r requirements.txt
	mkdir -p data/raw data/processed outputs/models outputs/midi outputs/logs

lint:
	flake8 src/ tests/ --max-line-length 120
	isort --check-only src/ tests/

format:
	black src/ tests/ --line-length 120
	isort src/ tests/

test:
	pytest tests/ -v

clean:
	rm -rf __pycache__ .pytest_cache outputs/logs/*
	find . -name "*.pyc" -delete
