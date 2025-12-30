"""Tests for mathematical utility functions."""
import numpy as np
import pytest
from fusang import comb_math, nlargest_indices


class TestCombMath:
    """Tests for comb_math function."""

    def test_basic_combinations(self):
        """Test basic combination calculations."""
        assert comb_math(5, 2) == 10
        assert comb_math(5, 3) == 10
        assert comb_math(4, 2) == 6
        assert comb_math(10, 1) == 10
        assert comb_math(10, 10) == 1

    def test_edge_cases(self):
        """Test edge cases."""
        assert comb_math(0, 0) == 1
        assert comb_math(5, 0) == 1
        assert comb_math(5, 5) == 1


class TestNLargestIndices:
    """Tests for nlargest_indices function."""

    def test_basic_functionality_2d(self):
        """Test basic functionality with 2D array (as used in actual code)."""
        # nlargest_indices is used with 2D arrays in select_mask_node_pair
        arr = np.array([[0.1, 0.2, 0.9], [0.8, 0.1, 0.1], [0.1, 0.1, 0.8]])
        row_indices, col_indices = nlargest_indices(arr, 2)
        # Should return indices where value >= threshold
        assert len(row_indices) > 0
        assert len(col_indices) > 0

    def test_1d_array(self):
        """Test with 1D array (returns single array in tuple)."""
        arr = np.array([1, 5, 3, 9, 2, 8, 4])
        result = nlargest_indices(arr, 3)
        # For 1D array, np.where returns (array([indices]),)
        assert isinstance(result, tuple)
        assert len(result) == 1
        indices = result[0]
        assert len(indices) >= 3

    def test_2d_array_usage(self):
        """Test with 2D array matching actual usage pattern."""
        # Simulate select_distribution from select_mask_node_pair
        arr = np.random.rand(10, 3)
        arr[0, 0] = 0.95  # High value
        arr[1, 0] = 0.95
        row_indices, col_indices = nlargest_indices(arr, 2)
        assert len(row_indices) > 0
        assert len(col_indices) > 0

