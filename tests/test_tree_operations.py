"""Tests for tree operation functions."""
import pytest
from ete3 import Tree
from fusang import get_modify_tree, transform_str, tree_from_quartet


class TestGetModifyTree:
    """Tests for get_modify_tree function."""

    def test_add_node_same_edge(self):
        """Test adding node when edge_0 == edge_1."""
        tree = tree_from_quartet('ABCD')
        root = tree.get_tree_root()
        
        modified = get_modify_tree(tree, root.name, root.name, 'NEW_NODE')
        
        # Should add new node as child
        leaves = [leaf.name for leaf in modified.get_leaves()]
        assert 'NEW_NODE' in leaves

    def test_add_node_different_edges(self):
        """Test adding node between different edges."""
        tree = tree_from_quartet('ABCD')
        root = tree.get_tree_root()
        child = root.children[0]
        
        modified = get_modify_tree(tree, root.name, child.name, 'NEW_NODE')
        
        # Should have new node in tree
        leaves = [leaf.name for leaf in modified.get_leaves()]
        assert 'NEW_NODE' in leaves
        assert len(leaves) == 5  # Original 4 + new node


class TestTransformStr:
    """Tests for transform_str function."""

    def test_basic_transformation(self):
        """Test basic string transformation."""
        taxa_name = {0: 'Species1', 1: 'Species2', 2: 'Species3'}
        taxa_num = 3
        
        # Create string with internal IDs (Chinese characters)
        str_a = '(' + chr(ord(u'\u4e00')) + ',' + chr(ord(u'\u4e00') + 1) + ')'
        result = transform_str(str_a, taxa_name, taxa_num)
        
        assert 'Species1' in result or 'Species2' in result

    def test_mixed_content(self):
        """Test transformation with mixed content."""
        taxa_name = {0: 'A', 1: 'B'}
        taxa_num = 2
        
        str_a = '(' + chr(ord(u'\u4e00')) + ',internal_node_1)'
        result = transform_str(str_a, taxa_name, taxa_num)
        
        assert 'A' in result
        assert 'internal_node_1' in result

    def test_no_transformation(self):
        """Test string with no internal IDs."""
        taxa_name = {0: 'A', 1: 'B'}
        taxa_num = 2
        
        str_a = '(internal_node_1,internal_node_2)'
        result = transform_str(str_a, taxa_name, taxa_num)
        
        assert result == str_a

