"""Process-based parallel map with progress reporting.

``parallel_map`` runs a picklable function over an iterable using a process
pool, returning results in input order even though execution is unordered. It
is the single parallel primitive in the package; higher-level batch helpers are
built on top of it. With ``n_jobs == 1`` it runs serially with no pool.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, Iterable, List, Optional, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def _resolve_workers(n_jobs: int) -> int:
    if n_jobs is None or n_jobs == 0:
        return 1
    if n_jobs < 0:
        import os

        return max(1, (os.cpu_count() or 1))
    return n_jobs


def parallel_map(
    func: Callable[[T], R],
    items: Iterable[T],
    n_jobs: int = 1,
    progress: bool = False,
    desc: Optional[str] = None,
) -> List[R]:
    """Apply ``func`` to each item, optionally in parallel.

    Parameters
    ----------
    func : callable
        A picklable function mapping one item to one result.
    items : iterable
        Input items.
    n_jobs : int, default 1
        Number of worker processes. ``1`` runs serially (no pool); ``-1`` uses
        all available CPUs.
    progress : bool, default False
        Display a tqdm progress bar.
    desc : str, optional
        Progress bar label.

    Returns
    -------
    list
        Results in the same order as ``items``.
    """
    items = list(items)
    n = len(items)

    bar = None
    if progress:
        from tqdm import tqdm

        bar = tqdm(total=n, desc=desc)

    workers = _resolve_workers(n_jobs)

    try:
        if workers == 1:
            results: List[R] = []
            for item in items:
                results.append(func(item))
                if bar is not None:
                    bar.update(1)
            return results

        results = [None] * n
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(func, item): i for i, item in enumerate(items)}
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()
                if bar is not None:
                    bar.update(1)
        return results
    finally:
        if bar is not None:
            bar.close()
