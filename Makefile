.PHONY: check test

PYTHON ?= python3

check:
	$(PYTHON) -m compileall -q evimem gavln llava trl vggt tests scripts
	bash -n scripts/create_env.sh scripts/download_mp3d.sh scripts/stage_assets.sh scripts/run_gavln_baseline.sh scripts/run_evimem.sh
	$(PYTHON) -m gavln.gavln_eval --help >/dev/null

test: check
	$(PYTHON) -m pytest -q
