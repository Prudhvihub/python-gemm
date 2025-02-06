import numpy as np
import pytest
from gemm import GEMM

def test_gemm_basic():
    gemm = GEMM(block_size=2)
    A = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ], dtype=float)
    B = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ], dtype=float)
    
    result = gemm.multiply(A, B)
    expected = np.dot(A, B)
    
    assert np.allclose(result, expected)

def test_gemm_invalid_dimensions():
    gemm = GEMM(block_size=4)
    A = np.random.rand(5, 5)  # Not divisible by block_size
    B = np.random.rand(5, 5)
    
    with pytest.raises(ValueError):
        gemm.multiply(A, B)

def test_gemm_large_matrices():
    gemm = GEMM(block_size=4)
    size = 128
    A = np.random.rand(size, size)
    B = np.random.rand(size, size)
    
    result = gemm.multiply(A, B)
    expected = np.dot(A, B)
    
    assert np.allclose(result, expected) 