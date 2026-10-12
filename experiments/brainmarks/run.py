"""Run brainmarks-smri (MedARC-AI/brainmarks-smri) with the nanobrain adapter registered, using
nanobrain's venv and an unmodified brainmarks clone on PYTHONPATH.

DATA_ROOT=/data/connor/brain_datasets/datasets PYTHONPATH=/admin/home/dojo/brainmarks-smri/src \
    .venv/bin/python experiments/brainmarks/run.py nanobrain ixi_age \
    --overrides model_kwargs.checkpoint=<checkpoint> output_dir=<run dir> name=brainmarks
"""

import nanobrain_adapter  # noqa: F401  registers model "nanobrain"
from brainmarks_smri.run import main

if __name__ == "__main__":
    main()
