"""Tests for evaluation functions."""
import pytest
from fusang import calculate_rf_distance


class TestCalculateRFDistance:
    """Tests for calculate_rf_distance function."""

    def test_identical_trees(self):
        """Test RF distance between identical trees."""
        tree_str = "((A,B),(C,D));"
        rf_distance, normalized_rf, max_rf = calculate_rf_distance(
            tree_str, tree_str, unrooted=True
        )
        
        # Check that calculation succeeded
        assert rf_distance is not None, "RF distance calculation failed"
        assert rf_distance == 0
        assert normalized_rf == 0.0
        assert max_rf > 0

    def test_different_topologies(self):
        """Test RF distance between different topologies."""
        tree1 = "((A,B),(C,D));"
        tree2 = "((A,C),(B,D));"
        
        rf_distance, normalized_rf, max_rf = calculate_rf_distance(
            tree1, tree2, unrooted=True
        )
        
        # Check that calculation succeeded
        assert rf_distance is not None, "RF distance calculation failed"
        assert rf_distance > 0
        assert 0.0 <= normalized_rf <= 1.0
        assert max_rf > 0

    def test_invalid_tree(self):
        """Test handling of invalid tree strings."""
        tree1 = "((A,B),(C,D));"
        tree2 = "invalid_tree_string"
        
        rf_distance, normalized_rf, max_rf = calculate_rf_distance(
            tree1, tree2, unrooted=True
        )
        
        assert rf_distance is None
        assert normalized_rf is None
        assert max_rf is None

    def test_unrooted_vs_rooted(self):
        """Test that unrooted parameter works."""
        tree1 = "((A,B),(C,D));"
        tree2 = "((A,B),(C,D));"
        
        rf_unrooted, norm_rf_unrooted, max_rf_unrooted = calculate_rf_distance(
            tree1, tree2, unrooted=True
        )
        rf_rooted, norm_rf_rooted, max_rf_rooted = calculate_rf_distance(
            tree1, tree2, unrooted=False
        )
        
        # Check that calculations succeeded
        assert rf_unrooted is not None, "Unrooted RF distance calculation failed"
        assert rf_rooted is not None, "Rooted RF distance calculation failed"
        # For identical trees, both should be 0
        assert rf_unrooted == 0
        assert rf_rooted == 0

