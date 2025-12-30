"""Tests for data processing functions."""
import functools
from io import StringIO
import numpy as np
import pytest
from fusang import cmp, _process_alignment, get_numpy, initialize_quartet_data


class TestCmp:
    """Tests for cmp function."""

    def test_basic_comparison(self):
        """Test basic numeric comparison."""
        assert cmp('1', '2') == -1
        assert cmp('2', '1') == 1
        assert cmp('5', '5') == 0

    def test_larger_numbers(self):
        """Test with larger numbers."""
        assert cmp('10', '2') == 1
        assert cmp('100', '99') == 1
        assert cmp('50', '50') == 0

    def test_used_in_sorting(self):
        """Test that cmp works with sorted."""
        data = ['10', '2', '5', '1', '20']
        sorted_data = sorted(data, key=functools.cmp_to_key(cmp))
        assert sorted_data == ['1', '2', '5', '10', '20']


class TestProcessAlignment:
    """Tests for _process_alignment function."""

    def test_basic_fasta_parsing(self):
        """Test basic FASTA parsing."""
        fasta_content = """>0
ATCG
>1
GCTA
"""
        aln = StringIO(fasta_content)
        result = _process_alignment(aln)
        
        assert result.shape[0] == 1  # Batch dimension
        assert result.shape[1] == 2  # Two sequences
        # Check encoding: A=0, T=1, C=2, G=3
        assert result[0, 0, 0] == 0  # First sequence, first position: A
        assert result[0, 0, 1] == 1  # First sequence, second position: T

    def test_gap_handling(self):
        """Test gap and unknown base handling."""
        fasta_content = """>0
AT-CG
>1
GCTAN
"""
        aln = StringIO(fasta_content)
        result = _process_alignment(aln)
        
        # Gaps and N should be encoded as 4
        assert result[0, 0, 2] == 4  # Gap
        assert result[0, 1, 4] == 4  # N

    def test_unknown_bases(self):
        """Test handling of ambiguous bases."""
        fasta_content = """>0
ATCG
>1
GCTR
"""
        aln = StringIO(fasta_content)
        result = _process_alignment(aln)
        
        # R should be encoded as 4
        assert result[0, 1, 3] == 4

    def test_sequence_padding(self):
        """Test that sequences are padded correctly."""
        from fusang import SEQUENCE_PADDING
        fasta_content = """>0
AT
>1
GC
"""
        aln = StringIO(fasta_content)
        result = _process_alignment(aln)
        
        # Should be padded to SEQUENCE_PADDING (14000)
        assert result.shape[2] == SEQUENCE_PADDING


class TestGetNumpy:
    """Tests for get_numpy function."""

    def test_file_like_object(self):
        """Test with file-like object."""
        fasta_content = """>0
ATCG
>1
GCTA
"""
        aln = StringIO(fasta_content)
        result = get_numpy(aln)
        
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 1

    def test_file_path(self, tmp_path):
        """Test with file path."""
        fasta_file = tmp_path / "test.fasta"
        fasta_content = """>0
ATCG
>1
GCTA
"""
        fasta_file.write_text(fasta_content)
        
        result = get_numpy(str(fasta_file))
        
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 1


class TestInitializeQuartetData:
    """Tests for initialize_quartet_data function."""

    def test_basic_initialization(self):
        """Test basic quartet data initialization."""
        taxa_num = 5
        (start_end_list, comb_of_id, leave_node_name, leave_node_comb_name,
         dic_for_leave_node_comb_name, internal_node_name_pool) = initialize_quartet_data(taxa_num)
        
        # Check start_end_list
        assert len(start_end_list) >= taxa_num
        assert start_end_list[3] is not None
        
        # Check combinations
        assert len(comb_of_id) == 5  # C(5, 4) = 5
        
        # Check leave node names
        assert len(leave_node_name) == taxa_num
        
        # Check dictionary
        assert len(dic_for_leave_node_comb_name) == len(comb_of_id)
        
        # Check internal node name pool
        assert len(internal_node_name_pool) > 0

    def test_small_taxa_number(self):
        """Test with small number of taxa."""
        taxa_num = 4
        (start_end_list, comb_of_id, leave_node_name, leave_node_comb_name,
         dic_for_leave_node_comb_name, internal_node_name_pool) = initialize_quartet_data(taxa_num)
        
        # C(4, 4) = 1
        assert len(comb_of_id) == 1
        assert len(leave_node_name) == 4

    def test_larger_taxa_number(self):
        """Test with larger number of taxa."""
        taxa_num = 10
        (start_end_list, comb_of_id, leave_node_name, leave_node_comb_name,
         dic_for_leave_node_comb_name, internal_node_name_pool) = initialize_quartet_data(taxa_num)
        
        # C(10, 4) = 210
        assert len(comb_of_id) == 210
        assert len(leave_node_name) == 10

