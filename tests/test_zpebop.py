"""Unit tests for ZPEBOP package."""

import pytest
import numpy as np

from zpebop.constants import (
    HARTREE_TO_KCAL,
    KCAL_TO_HARTREE,
    SUPPORTED_ELEMENTS,
    ELEMENT_TO_INDEX,
    N_ELEMENTS,
    BETA_V1_BOND,
    BETA_V1_ANTI,
    BETA_BOND,
    BETA_ANTI,
    KAPPA_BOND,
    KAPPA_ANTI,
    HAS_KAPPA,
)
from zpebop.models.base import ZPEResult, BondEnergies
from zpebop.parser import compute_distance_matrix


class TestConstants:
    """Tests for constants module."""
    
    def test_hartree_to_kcal_value(self):
        """Test that HARTREE_TO_KCAL has correct value."""
        assert abs(HARTREE_TO_KCAL - 627.5096) < 0.0001
    
    def test_hartree_conversion_roundtrip(self):
        """Test that converting back and forth gives original value."""
        value = 1.0
        converted = value * HARTREE_TO_KCAL * KCAL_TO_HARTREE
        assert abs(converted - value) < 1e-10
    
    def test_supported_elements_count(self):
        """Test that we have 18 supported elements."""
        assert len(SUPPORTED_ELEMENTS) == 18
        assert N_ELEMENTS == 18
    
    def test_element_index_mapping(self):
        """Test element to index mapping."""
        assert ELEMENT_TO_INDEX['H'] == 0
        assert ELEMENT_TO_INDEX['C'] == 5
        assert ELEMENT_TO_INDEX['O'] == 7


class TestParameterArrays:
    """Tests for pre-computed parameter arrays."""
    
    def test_beta_v1_shape(self):
        """Test ZPEBOP-1 BETA array shapes."""
        assert BETA_V1_BOND.shape == (N_ELEMENTS, N_ELEMENTS)
        assert BETA_V1_ANTI.shape == (N_ELEMENTS, N_ELEMENTS)
    
    def test_beta_v2_shapes(self):
        """Test ZPEBOP-2 parameter array shapes."""
        assert BETA_BOND.shape == (N_ELEMENTS, N_ELEMENTS)
        assert BETA_ANTI.shape == (N_ELEMENTS, N_ELEMENTS)
        assert KAPPA_BOND.shape == (N_ELEMENTS, N_ELEMENTS)
        assert KAPPA_ANTI.shape == (N_ELEMENTS, N_ELEMENTS)
    
    def test_arrays_read_only(self):
        """Test that parameter arrays are read-only."""
        assert not BETA_V1_BOND.flags.writeable
        assert not BETA_V1_ANTI.flags.writeable
        assert not BETA_BOND.flags.writeable
        assert not KAPPA_BOND.flags.writeable
    
    def test_cc_parameters_exist_v1(self):
        """Test that C-C parameters exist for ZPEBOP-1."""
        idx_c = ELEMENT_TO_INDEX['C']
        assert not np.isnan(BETA_V1_BOND[idx_c, idx_c])
    
    def test_cc_parameters_exist_v2(self):
        """Test that C-C parameters exist for ZPEBOP-2."""
        idx_c = ELEMENT_TO_INDEX['C']
        assert not np.isnan(BETA_BOND[idx_c, idx_c])
        assert HAS_KAPPA[idx_c, idx_c]
    
    def test_hh_has_no_kappa(self):
        """Test that H-H has no kappa parameter."""
        idx_h = ELEMENT_TO_INDEX['H']
        assert np.isnan(KAPPA_BOND[idx_h, idx_h])
        assert not HAS_KAPPA[idx_h, idx_h]
    
    def test_symmetry_v1(self):
        """Test that ZPEBOP-1 BETA is symmetric."""
        idx_c = ELEMENT_TO_INDEX['C']
        idx_h = ELEMENT_TO_INDEX['H']
        assert BETA_V1_BOND[idx_c, idx_h] == BETA_V1_BOND[idx_h, idx_c]
        assert BETA_V1_ANTI[idx_c, idx_h] == BETA_V1_ANTI[idx_h, idx_c]
    
    def test_symmetry_v2(self):
        """Test that ZPEBOP-2 parameters are symmetric."""
        idx_c = ELEMENT_TO_INDEX['C']
        idx_h = ELEMENT_TO_INDEX['H']
        assert BETA_BOND[idx_c, idx_h] == BETA_BOND[idx_h, idx_c]


class TestDistanceMatrix:
    """Tests for distance matrix computation."""
    
    def test_compute_distance_matrix_simple(self):
        """Test distance matrix for simple case."""
        coords = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        dist = compute_distance_matrix(coords)
        
        assert abs(dist[1, 0] - 1.0) < 1e-10
        assert abs(dist[2, 0] - 1.0) < 1e-10
        assert abs(dist[2, 1] - np.sqrt(2)) < 1e-10
    
    def test_compute_distance_matrix_lower_triangular(self):
        """Test that distance matrix is lower triangular."""
        coords = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        ])
        dist = compute_distance_matrix(coords)
        
        assert dist[0, 1] == 0.0
        assert dist[1, 0] > 0


class TestZPEResult:
    """Tests for ZPEResult dataclass."""
    
    def test_n_atoms_property(self):
        """Test n_atoms property."""
        atoms = np.array(['C', 'H', 'H'])
        result = ZPEResult(
            total_zpe=10.0,
            two_body=np.zeros((3, 3)),
            three_body_decomp=np.zeros((3, 3)),
            atoms=atoms,
            model='zpebop1',
            units='kcal/mol'
        )
        assert result.n_atoms == 3
    
    def test_gross_property(self):
        """Test gross property returns two_body."""
        two_body = np.array([[0, 0], [1, 0]])
        result = ZPEResult(
            total_zpe=1.0,
            two_body=two_body,
            three_body_decomp=np.zeros((2, 2)),
            atoms=np.array(['H', 'H']),
            model='zpebop1',
            units='kcal/mol'
        )
        assert np.array_equal(result.gross, two_body)
    
    def test_net_property(self):
        """Test net property returns two_body + three_body."""
        two_body = np.array([[0, 0], [1, 0]])
        three_body = np.array([[0, 0], [0.5, 0]])
        result = ZPEResult(
            total_zpe=1.5,
            two_body=two_body,
            three_body_decomp=three_body,
            atoms=np.array(['H', 'H']),
            model='zpebop2',
            units='kcal/mol'
        )
        expected = two_body + three_body
        assert np.array_equal(result.net, expected)
    
    def test_model_attribute(self):
        """Test model attribute."""
        result = ZPEResult(
            total_zpe=1.0,
            two_body=np.zeros((2, 2)),
            three_body_decomp=np.zeros((2, 2)),
            atoms=np.array(['H', 'H']),
            model='zpebop2',
            units='kcal/mol'
        )
        assert result.model == 'zpebop2'


class TestBondEnergies:
    """Tests for BondEnergies dataclass."""
    
    def test_dataclass_creation(self):
        """Test BondEnergies can be created."""
        gross = np.array([[0, 0], [1, 0]])
        net = np.array([[0, 0], [1.5, 0]])
        composite = np.array([[0, 1], [1.5, 0]])
        
        be = BondEnergies(
            gross=gross,
            net=net,
            composite=composite,
            units='kcal/mol'
        )
        
        assert be.units == 'kcal/mol'
        assert np.array_equal(be.gross, gross)
        assert np.array_equal(be.net, net)


class TestIsotope:
    """Tests for isotope correction functionality."""
    
    def test_atomic_masses_exist(self):
        """Test that atomic masses are defined for all supported elements."""
        from zpebop.constants import ATOMIC_MASSES, SUPPORTED_ELEMENTS
        for elem in SUPPORTED_ELEMENTS:
            assert elem in ATOMIC_MASSES
            assert ATOMIC_MASSES[elem] > 0
    
    def test_common_isotopes_exist(self):
        """Test that common isotopes are defined."""
        from zpebop.constants import COMMON_ISOTOPES
        assert 'H' in COMMON_ISOTOPES
        assert 'D' in COMMON_ISOTOPES['H']
        assert abs(COMMON_ISOTOPES['H']['D'] - 2.014) < 0.01
    
    def test_isotope_result_properties(self):
        """Test IsotopeZPEResult dataclass properties."""
        from zpebop.models.base import IsotopeZPEResult
        
        two_body = np.array([[0, 0], [10, 0]])
        three_body = np.array([[0, 0], [1, 0]])
        correction = np.array([[1, 1], [0.7, 1]])
        
        result = IsotopeZPEResult(
            total_zpe=7.7,
            total_zpe_normal=11.0,
            two_body=two_body * correction,
            two_body_normal=two_body,
            three_body_decomp=three_body * correction,
            three_body_decomp_normal=three_body,
            correction_factors=correction,
            atoms=np.array(['H', 'C']),
            isotopes={1: 2.014},
            model='zpebop1',
            units='kcal/mol'
        )
        
        assert result.n_atoms == 2
        assert result.zpe_ratio == 7.7 / 11.0
        assert result.zpe_difference == 11.0 - 7.7
    
    def test_isotope_correction_factor(self):
        """Test that isotope correction factor is physically correct."""
        # C-H vs C-D
        m_C = 12.0
        m_H = 1.00782503207
        m_D = 2.01410177812
        
        mu_CH = (m_C * m_H) / (m_C + m_H)
        mu_CD = (m_C * m_D) / (m_C + m_D)
        
        expected_correction = np.sqrt(mu_CH / mu_CD)
        
        # Should be approximately 0.7342
        assert abs(expected_correction - 0.7342) < 0.001
