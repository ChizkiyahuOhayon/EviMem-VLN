.PHONY: check test

PYTHON ?= python3

check:
	$(PYTHON) -m compileall -q src tests scripts
	bash -n scripts/create_env.sh scripts/bootstrap_ga_vln.sh scripts/download_mp3d.sh scripts/stage_assets.sh scripts/run_gavln_baseline.sh

test: check
	$(PYTHON) -m unittest discover -s tests -v
