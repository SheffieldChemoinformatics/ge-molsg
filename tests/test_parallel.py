"""Test cases for the parallel map primitive."""
from ge_molsg import parallel_map


def _square(x):
    return x * x


def test_parallel_map_serial_preserves_order():
    """Serial execution returns results in input order."""
    result = parallel_map(_square, [1, 2, 3, 4], n_jobs=1)
    assert result == [1, 4, 9, 16]


def test_parallel_map_parallel_preserves_order():
    """Parallel execution returns results in input order."""
    result = parallel_map(_square, list(range(10)), n_jobs=2)
    assert result == [i * i for i in range(10)]


def test_parallel_map_empty_input():
    """An empty iterable yields an empty result list."""
    assert parallel_map(_square, [], n_jobs=2) == []
