# Convenience front door. Real logic lives in run_eval.sh / infra/*.sh / tools/.
.PHONY: help eval check entrypoint container package verify figures clean

help:
	@echo "make eval        - run the official 14-row matrix in the container (run_eval.sh)"
	@echo "make check       - verify_baseline + sync_entrypoint --check + check_validity"
	@echo "make entrypoint  - regenerate torch_transformer_benchmark.py from benchmark.py"
	@echo "make figures     - regenerate assets/*.svg for the README"
	@echo "make container   - build /scratch/kernel.sif from infra/apptainer/kernel.def"
	@echo "make package     - clean-tree checks + git archive -> dist/techjam2_<ver>.tar.gz"
	@echo "make verify      - smoke-test the newest dist/ tarball"
	@echo "make clean       - remove dist/"

eval:
	./run_eval.sh

check:
	python3 tools/verify_baseline.py
	python3 tools/sync_entrypoint.py --check
	python3 tools/check_validity.py benchmark.py

entrypoint:
	python3 tools/sync_entrypoint.py

figures:
	python3 tools/make_figures.py

container:
	bash infra/apptainer/build.sh

package:
	bash infra/package.sh

verify:
	bash infra/verify_submission.sh $(shell ls -t dist/techjam2_*.tar.gz 2>/dev/null | head -1)

clean:
	rm -rf dist/
