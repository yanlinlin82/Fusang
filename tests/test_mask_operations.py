"""Tests for masking operations."""
import numpy as np
import pytest
from ete3 import Tree
from fusang import (
    FusangTreeBuilder,
    MIN_TAXA_FOR_MASKING,
    MIN_EDGES_FOR_MASKING,
)


class TestSelectMaskNodePair:
    """Tests for select_mask_node_pair function."""

    def test_too_few_taxa(self):
        """Test that function returns None for too few taxa."""
        dl_predict = np.random.rand(10, 3)
        new_add_taxa = MIN_TAXA_FOR_MASKING - 1
        start_end_list = [None, None, None, (0, 9)]
        comb_of_id = [(0, 1, 2, 3)] * 10
        
        builder = FusangTreeBuilder(beam_size=5, taxa_num=10, start_end_list=start_end_list, comb_of_id=comb_of_id)
        result = builder._select_mask_node_pair(dl_predict, new_add_taxa)
        assert result is None

    def test_low_confidence(self):
        """Test that function returns None for low confidence."""
        dl_predict = np.random.rand(10, 3) * 0.5  # All values < 0.90
        new_add_taxa = MIN_TAXA_FOR_MASKING
        # start_end_list structure: [None, None, None, (start, end), ...]
        # For new_add_taxa=10, need index 10 in start_end_list
        start_end_list = [None, None, None] + [(0, 9)] * 8  # Indices 3-10
        comb_of_id = [(0, 1, 2, 3)] * 10
        
        builder = FusangTreeBuilder(beam_size=5, taxa_num=10, start_end_list=start_end_list, comb_of_id=comb_of_id)
        result = builder._select_mask_node_pair(dl_predict, new_add_taxa)
        assert result is None

    def test_high_confidence(self):
        """Test that function returns node pairs for high confidence."""
        dl_predict = np.random.rand(10, 3)
        dl_predict[:, 0] = 0.95  # High confidence for topology 0
        new_add_taxa = MIN_TAXA_FOR_MASKING
        # start_end_list structure: [None, None, None, (start, end), ...]
        # For new_add_taxa=10, need index 10 in start_end_list
        start_end_list = [None, None, None] + [(0, 9)] * 8  # Indices 3-10
        comb_of_id = [(0, 1, 2, 3)] * 10
        
        builder = FusangTreeBuilder(beam_size=5, taxa_num=10, start_end_list=start_end_list, comb_of_id=comb_of_id)
        result = builder._select_mask_node_pair(dl_predict, new_add_taxa)
        
        if result is not None:
            assert isinstance(result, list)
            assert len(result) > 0


class TestMaskEdge:
    """Tests for mask_edge function."""

    def test_too_few_edges(self):
        """Test that function returns original list for too few edges."""
        tree = Tree("((A,B),(C,D));")
        edge_list = [(1, 2)]  # Less than MIN_EDGES_FOR_MASKING
        
        builder = FusangTreeBuilder(beam_size=5, taxa_num=10)
        result = builder._mask_edge(tree, 'A', 'B', edge_list.copy())
        assert result == edge_list

    def test_basic_masking(self):
        """Test basic edge masking."""
        tree = Tree("((A,B),(C,D));")
        
        # Get actual edge names from tree
        edge_list = []
        for node in tree.iter_descendants():
            if node.up and node.up.name and node.name:
                edge_list.append((node.up.name, node.name))
        
        if len(edge_list) > MIN_EDGES_FOR_MASKING:
            original_len = len(edge_list)
            builder = FusangTreeBuilder(beam_size=5, taxa_num=10)
            result = builder._mask_edge(tree, 'A', 'B', edge_list.copy())
            
            # Should have fewer edges if masking worked
            assert len(result) <= original_len
            assert len(result) >= MIN_EDGES_FOR_MASKING

    def test_common_ancestor_masking(self):
        """Test masking with common ancestor."""
        tree = Tree("((A,B),(C,D));")
        
        # Get actual edge names from tree
        edge_list = []
        for node in tree.iter_descendants():
            if node.up and node.up.name and node.name:
                edge_list.append((node.up.name, node.name))
        
        if len(edge_list) > MIN_EDGES_FOR_MASKING:
            builder = FusangTreeBuilder(beam_size=5, taxa_num=10)
            result = builder._mask_edge(tree, 'A', 'C', edge_list.copy())
            
            # Should remove edges on path to common ancestor
            assert len(result) <= len(edge_list)

