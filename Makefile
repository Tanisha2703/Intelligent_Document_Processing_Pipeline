# Common tasks. On Windows run the commands directly, or use Git Bash with make installed.
.PHONY: install test lint format demo api clean

install:            ## runtime + test dependencies, and the linter
	pip install -r requirements.txt ruff

test:               ## 117 tests, about 10 seconds, no GPU and no network
	pytest -q

lint:               ## static checks and formatting check (the same as CI)
	ruff check .
	ruff format --check .

format:             ## apply the formatter
	ruff format .
	ruff check . --fix

demo:               ## Streamlit demo; the recorded examples need no GPU
	streamlit run app.py --server.address 127.0.0.1

api:                ## the HTTP service on localhost (set DOCINTEL_VLM_APP / DOCINTEL_DOCTYPE for the models)
	uvicorn "src.api.app:create_app" --factory --port 8000

clean:              ## remove caches
	find . -name __pycache__ -not -path "./data/*" -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache
