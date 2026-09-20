import numpy as np

from plotting import normalize_trace


def test_normalize_trace_uses_each_trace_own_range():
    first = normalize_trace(np.array([10.0, 15.0, 20.0]))
    second = normalize_trace(np.array([1000.0, 1500.0, 2000.0]))
    np.testing.assert_allclose(first, [0.0, 0.5, 1.0])
    np.testing.assert_allclose(second, [0.0, 0.5, 1.0])


def test_normalize_constant_trace_centers_it():
    scaled = normalize_trace(np.array([5.0, 5.0, 5.0]))
    np.testing.assert_allclose(scaled, [0.5, 0.5, 0.5])
