"""SynthGen: prior-sampled synthetic benchmark suite (decision tier 1).

Deterministic numpy-only SCM-prior datasets, calibrated to harvested public
dataset metadata (TabArena and CTR23 excluded).

Modified by the DarkoFit project from ChimeraBoost 0.15.0 commit 851ab7f.
See ``NOTICE`` and ``benchmarks/synthgen_darkofit_protocol.md``.
"""
from .api import (build_dataset, hash_dataset, key_for, make_builder,  # noqa: F401
                  parse_key, recipe_meta, sample_recipe, task_of)
from .recipe import VERSION  # noqa: F401
from .suites import SUITES, all_frozen_keys, frozen_keys  # noqa: F401
