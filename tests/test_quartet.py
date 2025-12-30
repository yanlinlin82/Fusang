"""Tests for quartet-related functions."""
import pytest
from fusang import (
    get_quartet_ID,
    get_topology_ID,
    get_current_topology_id,
    tree_from_quartet,
)


class TestGetQuartetID:
    """Tests for get_quartet_ID function."""

    def test_basic_functionality(self):
        """Test basic quartet ID generation."""
        quartet = ['A', 'B', 'C', 'D']
        result = get_quartet_ID(quartet)
        assert result == 'ABCD'
        assert result == get_quartet_ID(['D', 'C', 'B', 'A'])

    def test_unsorted_input(self):
        """Test that input order doesn't matter."""
        assert get_quartet_ID(['Z', 'A', 'B', 'C']) == 'ABCZ'
        assert get_quartet_ID(['C', 'B', 'A', 'Z']) == 'ABCZ'


class TestGetTopologyID:
    """Tests for get_topology_ID function."""

    def test_basic_functionality(self):
        """Test basic topology ID generation."""
        quartet = ['A', 'B', 'C', 'D']
        result = get_topology_ID(quartet)
        # Should be sorted first two + sorted last two
        assert result == 'ABCD'

    def test_different_order(self):
        """Test with different input order."""
        quartet = ['D', 'C', 'B', 'A']
        result = get_topology_ID(quartet)
        # get_topology_ID returns sorted(first_two) + sorted(last_two)
        # For ['D', 'C', 'B', 'A']: sorted(['D', 'C']) + sorted(['B', 'A']) = 'CD' + 'AB' = 'CDAB'
        assert result == 'CDAB'


class TestGetCurrentTopologyID:
    """Tests for get_current_topology_id function."""

    def test_topology_0(self):
        """Test topology ID 0 (pairs: 0-1 or 2-3)."""
        quart_key = 'ABCD'
        assert get_current_topology_id(quart_key, 'A', 'B') == 0
        assert get_current_topology_id(quart_key, 'B', 'A') == 0
        assert get_current_topology_id(quart_key, 'C', 'D') == 0
        assert get_current_topology_id(quart_key, 'D', 'C') == 0

    def test_topology_1(self):
        """Test topology ID 1 (pairs: 0-2 or 1-3)."""
        quart_key = 'ABCD'
        assert get_current_topology_id(quart_key, 'A', 'C') == 1
        assert get_current_topology_id(quart_key, 'C', 'A') == 1
        assert get_current_topology_id(quart_key, 'B', 'D') == 1
        assert get_current_topology_id(quart_key, 'D', 'B') == 1

    def test_topology_2(self):
        """Test topology ID 2 (pairs: 0-3 or 1-2)."""
        quart_key = 'ABCD'
        assert get_current_topology_id(quart_key, 'A', 'D') == 2
        assert get_current_topology_id(quart_key, 'D', 'A') == 2
        assert get_current_topology_id(quart_key, 'B', 'C') == 2
        assert get_current_topology_id(quart_key, 'C', 'B') == 2


class TestTreeFromQuartet:
    """Tests for tree_from_quartet function."""

    def test_basic_tree_creation(self):
        """Test basic tree creation from quartet."""
        quartet = 'ABCD'
        tree = tree_from_quartet(quartet)
        
        # Check that tree has correct structure
        leaves = [leaf.name for leaf in tree.get_leaves()]
        assert set(leaves) == {'A', 'B', 'C', 'D'}
        
        # Check that all distances are 0
        for node in tree.iter_descendants():
            assert node.dist == 0

    def test_tree_structure(self):
        """Test tree structure."""
        quartet = 'WXYZ'
        tree = tree_from_quartet(quartet)
        
        # Should have root with name internal_node_0
        assert tree.name == 'internal_node_0'
        
        # Should have 4 leaves
        leaves = tree.get_leaves()
        assert len(leaves) == 4

