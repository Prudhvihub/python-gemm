import numpy as np
from typing import List, Union, Optional
import time

class GEMM:
    def __init__(self, block_size: int = 4):
        """
        Initialize GEMM with configurable block size
        
        Args:
            block_size: Size of blocks for block matrix multiplication (default=4)
        """
        self.block_size = block_size

    def _validate_inputs(self, A: np.ndarray, B: np.ndarray) -> None:
        """
        Validate input matrices
        
        Args:
            A: First input matrix
            B: Second input matrix
            
        Raises:
            ValueError: If matrices are invalid
        """
        if A.shape[1] != B.shape[0]:
            raise ValueError("Matrix dimensions don't match for multiplication")
        if A.shape[0] % self.block_size != 0 or B.shape[1] % self.block_size != 0:
            raise ValueError(f"Matrix dimensions must be divisible by {self.block_size}")

    def _block_multiply(self, A_block: np.ndarray, B_block: np.ndarray) -> np.ndarray:
        """
        Multiply two blocks of matrices
        
        Args:
            A_block: Block from matrix A
            B_block: Block from matrix B
            
        Returns:
            Result of block multiplication
        """
        return np.dot(A_block, B_block)

    def multiply(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Perform optimized matrix multiplication
        
        Args:
            A: First input matrix
            B: Second input matrix
            
        Returns:
            Result of matrix multiplication
        """
        self._validate_inputs(A, B)
        
        m, k = A.shape
        k, n = B.shape
        
        # Initialize result matrix
        C = np.zeros((m, n))
        
        # Perform block matrix multiplication
        for i in range(0, m, self.block_size):
            for j in range(0, n, self.block_size):
                for p in range(0, k, self.block_size):
                    # Get blocks
                    A_block = A[i:i+self.block_size, p:p+self.block_size]
                    B_block = B[p:p+self.block_size, j:j+self.block_size]
                    
                    # Multiply blocks and add to result
                    C[i:i+self.block_size, j:j+self.block_size] += self._block_multiply(A_block, B_block)
        
        return C

def benchmark():
    """Run benchmark tests"""
    sizes = [128, 256, 512]
    gemm = GEMM(block_size=4)
    
    print("Running benchmarks...")
    print("-" * 50)
    print("Matrix Size | Time (s) | Verification")
    print("-" * 50)
    
    for size in sizes:
        # Generate random matrices
        A = np.random.rand(size, size)
        B = np.random.rand(size, size)
        
        # Time our implementation
        start_time = time.time()
        C_gemm = gemm.multiply(A, B)
        end_time = time.time()
        
        # Verify with numpy
        C_numpy = np.dot(A, B)
        is_correct = np.allclose(C_gemm, C_numpy, rtol=1e-5)
        
        print(f"{size}x{size}    | {end_time - start_time:.4f}  | {'✓' if is_correct else '✗'}")

if __name__ == "__main__":
    benchmark() 