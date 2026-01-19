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
ZPEBOP-1 model implementation.

ZPEBOP-1 computes zero-point vibrational energies using the
extended Hückel approximation with separate bonding and antibonding parameters:

    E(A-B) = 2 * β_AB * P_AB

where β_AB is an atom-pair-specific parameter (bonding if P > 0, antibonding if P < 0)
and P_AB is the Mulliken bond order.

References
----------
.. [1] Zulueta, B., Rude, C. D., Mangiardi, J. A., Petersson, G. A., & Keith, J. A.
       (2025). Zero-point energies from bond orders and populations relationships.
       The Journal of Chemical Physics, 162(8), 084102.
       https://doi.org/10.1063/5.0238831
"""

from typing import Tuple
import numpy as np

from ..constants import BETA_V1_BOND, BETA_V1_ANTI
from .base import ZPEResult

__all__ = ['compute_zpe_v1']


def compute_zpe_v1(atom_indices: np.ndarray,
                   bond_orders: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute ZPEBOP-1 ZPE using vectorized operations.
    
    ZPEBOP-1 uses the harmonic term with separate bonding/antibonding parameters:
        E = 2 * β * P
    where β is positive for bonding (P >= 0) and antibonding uses -β_anti (P < 0).
    
    The old code logic:
    - If P >= 0: E = 2 * β_bond * P (positive contribution)
    - If P < 0:  E = 2 * (-β_anti) * P = -2 * β_anti * P (positive since P < 0)
    
    Parameters
    ----------
    atom_indices : np.ndarray
        Array of element indices (from ELEMENT_TO_INDEX).
    bond_orders : np.ndarray
        Mulliken bond order matrix.
    
    Returns
    -------
    zpe : float
        Total zero-point energy in kcal/mol.
    two_body : np.ndarray
        Two-body (harmonic) contributions matrix.
    three_body_decomp : np.ndarray
        Zero matrix (ZPEBOP-1 has no three-body terms).
    """
    n = len(atom_indices)
    
    # Get lower triangular indices (pairs where i > j)
    row_idx, col_idx = np.tril_indices(n, k=-1)
    
    # Get element indices for all pairs
    idx1 = atom_indices[row_idx]
    idx2 = atom_indices[col_idx]
    
    # Get bond orders for all pairs
    bo_pairs = bond_orders[row_idx, col_idx]
    
    # Separate bonding (P >= 0) and antibonding (P < 0)
    is_bonding = bo_pairs >= 0
    
    # Get appropriate beta values
    # For bonding: use BETA_V1_BOND
    # For antibonding: use -BETA_V1_ANTI (the negation is part of the formula)
    beta = np.where(is_bonding,
                    BETA_V1_BOND[idx1, idx2],
                    -BETA_V1_ANTI[idx1, idx2])
    
    # Calculate energy: E = 2 * β * P
    # For bonding: β > 0, P >= 0 → E >= 0
    # For antibonding: β < 0 (negated), P < 0 → E > 0
    energy_pairs = 2.0 * beta * bo_pairs
    
    # Handle NaN (no parameters) -> 0
    energy_pairs = np.nan_to_num(energy_pairs, nan=0.0)
    
    # Build two-body matrix
    two_body = np.zeros((n, n), dtype=np.float64)
    two_body[row_idx, col_idx] = energy_pairs
    
    # ZPEBOP-1 has no three-body terms
    three_body_decomp = np.zeros((n, n), dtype=np.float64)
    
    # Total ZPE
    total_zpe = np.sum(two_body)
    
    return total_zpe, two_body, three_body_decomp
