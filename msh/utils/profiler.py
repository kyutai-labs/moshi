# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
import logging
import random
import time
import typing as tp

import dora
import torch


logger = logging.getLogger(__name__)


class Profiler:
    """Context manager wrapper for xformers profiler.
    """
    def __init__(self, module: torch.nn.Module, enabled: bool = False):
        self.profiler: tp.Optional[tp.Any] = None
        if enabled:
            from xformers.profiler import profile
            from xformers.profiler.api import PyTorchProfiler
            output_dir = dora.get_xp().folder / 'profiler_data'
            logger.info("Profiling activated, results with be saved to %s", output_dir)
            schedule = (
                (PyTorchProfiler, 6, 12),
            )
            self.profiler = profile(output_dir=output_dir, module=module, schedule=schedule)

    def step(self):
        if self.profiler is not None:
            self.profiler.step()  # type: ignore

    def __enter__(self):
        if self.profiler is not None:
            return self.profiler.__enter__()  # type: ignore

    def __exit__(self, exc_type, exc_value, exc_tb):
        if self.profiler is not None:
            return self.profiler.__exit__(exc_type, exc_value, exc_tb)  # type: ignore


class LowTechProfiler:
    """A lowtech profiler for debugging coarse level issues.
    In particular, it allows to divide the training over a batch in a number of *stages*,
    and to measure the time spent in each stage. The end of a stage is marked by a call to `step(name, sync)`,
    with `sync=True` (the default) if the stage requires a synchronization point (e.g. it is GPU bound).
    Use `sync=False` for instance for data-loading. When `sync=True`, and distributed is initialized,
    this will also trigger a barrier, and measure the delta between the sync point and the barrier,
    i.e. how much we were slowed down by other GPUs. In order to allow for this to run all the time,
    as we never know when the shit will hit the fan, we subsample how often we perform those actions
    (which are of course detrimental to the overal training speed).

    Args:
        enabled (boo): if False, deactivates everything.
        prefix (str): prefix for the metrics, default to `"ltp_"` (low tech profiler).
        proba (float): probability to introduce sync point and barriers, and to measure things.
    """

    def __init__(self, enabled: bool = True, prefix: str = 'ltp_', proba: float = 0.05) -> None:
        self.enabled = enabled
        self.prefix = prefix
        self.proba = proba
        self._last_time: tp.Optional[float] = None
        self._metrics: tp.Dict[str, tp.List[float]] = defaultdict(list)
        self._instant_metrics: tp.Dict[str, tp.List[float]] = defaultdict(list)
        self._rng = random.Random(4321)
        self._current_decision = False
        self._is_distributed = torch.distributed.is_initialized()

    def reset(self) -> None:
        """Reset the internal state, for instance when starting an epoch."""
        self._last_time = None
        self._metrics.clear()
        self._instant_metrics.clear()

    def reset_instant(self) -> None:
        """Reset instant metrics, to be called once you logged them (see `collect()` after).
        """
        self._instant_metrics.clear()

    def update_decision(self) -> None:
        """Should be called at the beginning of each batch, will update the decision to perform or not
        profiling for this batch. Note that the default state is to not profile until you call this function.
        Then a RNG synced with the same seed will make the decisions.
        """
        self._current_decision = self._rng.random() < self.proba

    def _add_point(self, name: str) -> None:
        # Add a data point to both the instant and global metrics.
        assert self._last_time is not None
        new_time = time.time()
        delta = new_time - self._last_time
        self._instant_metrics[name].append(delta)
        self._metrics[name].append(delta)
        self._last_time = new_time

    def step(self, name: str, sync: bool = True) -> None:
        """Call this at the end of a stage. Use `sync=True` if the stage is GPU bound."""
        if not self.enabled:
            return

        if not self._current_decision:
            # we still need to update the current timestamp.
            self._last_time = time.time()
            return

        if self._last_time is None:
            # Very first step, we will skip this step for the first time.
            self._last_time = time.time()
            return

        if sync:
            torch.cuda.synchronize()

        self._add_point(name)

        if sync and self._is_distributed:
            torch.distributed.barrier()
            self._add_point('ba_' + name)

    def collect(self, instant: bool = True) -> tp.Dict[str, float]:
        """Collect the metrics.

        Args:
            instant (bool): if True, returns the instant metrics. Instant metrics
                are flushed with `reset_instant()`. if False, return global metrics,
                both the mean and 90th percentile.
        """
        if not self.enabled:
            return {}

        if instant:
            metrics = self._instant_metrics
        else:
            metrics = self._metrics

        out = {}
        for key, values in metrics.items():
            values.sort()
            mean = sum(values) / len(values)
            out[self.prefix + key] = mean
            if not instant:
                # Most likely for instant, and with low proba we will have a single value, not worth it.
                for q in [90]:
                    idx = int((q * len(values) / 100))
                    out[self.prefix + key + f'_q{q}'] = values[idx]
        return out
