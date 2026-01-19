# MIT License
#
# Copyright (c) 2026, Barbaro Zulueta
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
ZPEBOP-2 model implementation.

ZPEBOP-2 computes zero-point vibrational energies using three terms:

1. Harmonic (extended Hückel):
    E_harm(A-B) = 2 * β_AB * |P_AB|

2. Anharmonic (short-range):
    E_anharm(A-B) = A * exp(-ζ * (R - R₀))

3. Three-body coupling:
    E_3body(i,j,k) = κ_ij * κ_ik * κ_jk * 2|P_ij| * 2|P_ik| * 2|P_jk| * cos_ij * cos_ik * cos_jk

Gross energy = E_harm + E_anharm (two-body only)
Net energy = Gross + E_3body (includes three-body)

References
----------
.. [1] Zulueta, B., Rude, C. D., Mangiardi, J. A., Petersson, G. A., & Keith, J. A.
       (2025). Zero-point energies from bond orders and populations relationships.
       The Journal of Chemical Physics, 162(8), 084102.
       https://doi.org/10.1063/5.0238831
"""

from typing import Tuple
import numpy as np

from ..constants import (
    BETA_BOND,
    BETA_ANTI,
    PRE_EXP,
    ZETA,
    R_PARAM,
    KAPPA_BOND,
    KAPPA_ANTI,
    HAS_KAPPA,
)
from .base import ZPEResult

__all__ = ['compute_zpe_v2']


def compute_zpe_v2(atom_indices: np.ndarray,
                   bond_orders: np.ndarray,
                   distance_matrix: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute ZPEBOP-2 ZPE using vectorized operations.
    
    ZPEBOP-2 uses harmonic + anharmonic + three-body terms.
    
    Parameters
    ----------
    atom_indices : np.ndarray
        Array of element indices (from ELEMENT_TO_INDEX).
    bond_orders : np.ndarray
        Mulliken bond order matrix.
    distance_matrix : np.ndarray
        Interatomic distance matrix (lower triangular).
    
    Returns
    -------
    zpe : float
        Total zero-point energy in kcal/mol.
    two_body : np.ndarray
        Two-body (harmonic + anharmonic) contributions matrix.
    three_body_decomp : np.ndarray
        Three-body contributions decomposed to pairs.
    """
    n = len(atom_indices)
    
    # Get lower triangular indices (pairs where i > j)
    row_idx, col_idx = np.tril_indices(n, k=-1)
    
    # Get element indices for all pairs
    idx1 = atom_indices[row_idx]
    idx2 = atom_indices[col_idx]
    
    # Get bond orders and distances for all pairs
    bo_pairs = bond_orders[row_idx, col_idx]
    dist_pairs = distance_matrix[row_idx, col_idx]
    
    # Determine bonding vs antibonding (antibonding if bo < 0 and n > 2)
    is_anti = (bo_pairs < 0) & (n > 2)
    
    # =========================================================================
    # Two-body: Harmonic term (vectorized)
    # =========================================================================
    beta = np.where(is_anti, BETA_ANTI[idx1, idx2], BETA_BOND[idx1, idx2])
    harmonic_pairs = 2.0 * beta * np.abs(bo_pairs)
    # Handle NaN (no parameters) -> 0
    harmonic_pairs = np.nan_to_num(harmonic_pairs, nan=0.0)
    
    # =========================================================================
    # Two-body: Anharmonic term (vectorized)
    # =========================================================================
    pre_exp = PRE_EXP[idx1, idx2]
    zeta = ZETA[idx1, idx2]
    r_param = R_PARAM[idx1, idx2]
    
    # Compute anharmonic: A * exp(-zeta * (R - R0))
    # Only where all parameters are defined (not NaN)
    has_anharm = ~(np.isnan(pre_exp) | np.isnan(zeta) | np.isnan(r_param))
    anharmonic_pairs = np.zeros(len(row_idx), dtype=np.float64)
    anharmonic_pairs[has_anharm] = (
        pre_exp[has_anharm] * 
        np.exp(-zeta[has_anharm] * (dist_pairs[has_anharm] - r_param[has_anharm]))
    )
    
    # =========================================================================
    # Build two-body matrix
    # =========================================================================
    two_body = np.zeros((n, n), dtype=np.float64)
    two_body_pairs = harmonic_pairs + anharmonic_pairs
    two_body[row_idx, col_idx] = two_body_pairs
    
    # =========================================================================
    # Three-body coupling
    # =========================================================================
    three_body_decomp = np.zeros((n, n), dtype=np.float64)
    three_body_total = 0.0
    
    # Iterate over all triplets (i > j > k)
    for i in range(2, n):
        for j in range(1, i):
            for k in range(j):
                # Check if all three pairs have kappa parameters
                idx_i, idx_j, idx_k = atom_indices[i], atom_indices[j], atom_indices[k]
                
                if not (HAS_KAPPA[idx_i, idx_j] and 
                        HAS_KAPPA[idx_i, idx_k] and 
                        HAS_KAPPA[idx_j, idx_k]):
                    continue
                
                # Get bond orders
                bo_ij = bond_orders[i, j]
                bo_ik = bond_orders[i, k]
                bo_jk = bond_orders[j, k]
                
                # Determine bonding/antibonding for each pair
                is_anti_ij = (bo_ij < 0) and (n > 2)
                is_anti_ik = (bo_ik < 0) and (n > 2)
                is_anti_jk = (bo_jk < 0) and (n > 2)
                
                # Get kappa parameters
                kappa_ij = KAPPA_ANTI[idx_i, idx_j] if is_anti_ij else KAPPA_BOND[idx_i, idx_j]
                kappa_ik = KAPPA_ANTI[idx_i, idx_k] if is_anti_ik else KAPPA_BOND[idx_i, idx_k]
                kappa_jk = KAPPA_ANTI[idx_j, idx_k] if is_anti_jk else KAPPA_BOND[idx_j, idx_k]
                
                # Skip if any kappa is NaN
                if np.isnan(kappa_ij) or np.isnan(kappa_ik) or np.isnan(kappa_jk):
                    continue
                
                # Get distances
                R_ij = distance_matrix[i, j]
                R_ik = distance_matrix[i, k]
                R_jk = distance_matrix[j, k]
                
                # Compute cosines using law of cosines
                cos_ij = np.clip((R_ik**2 + R_jk**2 - R_ij**2) / (2 * R_ik * R_jk), -1, 1)
                cos_ik = np.clip((R_ij**2 + R_jk**2 - R_ik**2) / (2 * R_ij * R_jk), -1, 1)
                cos_jk = np.clip((R_ij**2 + R_ik**2 - R_jk**2) / (2 * R_ij * R_ik), -1, 1)
                
                # Three-body energy
                kappa_prod = kappa_ij * kappa_ik * kappa_jk
                bo_prod = 2 * np.abs(bo_ij) * 2 * np.abs(bo_ik) * 2 * np.abs(bo_jk)
                cos_prod = cos_ij * cos_ik * cos_jk
                
                three_body_value = kappa_prod * bo_prod * cos_prod
                three_body_total += three_body_value
                
                # Decompose to two-body contributions
                e_ij = two_body[i, j]
                e_ik = two_body[i, k]
                e_jk = two_body[j, k]
                sum_two = e_ij + e_ik + e_jk
                
                # Proportional decomposition
                three_body_decomp[i, j] += three_body_value * (e_ij / sum_two)
                three_body_decomp[i, k] += three_body_value * (e_ik / sum_two)
                three_body_decomp[j, k] += three_body_value * (e_jk / sum_two)
    
    # Total ZPE
    total_zpe = np.sum(two_body) + three_body_total
    
    return total_zpe, two_body, three_body_decomp
