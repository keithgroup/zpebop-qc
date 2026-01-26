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
Base classes and dataclasses shared by ZPEBOP models.
"""

from dataclasses import dataclass
from typing import Dict
import numpy as np

__all__ = ['ZPEResult', 'BondEnergies']


@dataclass
class BondEnergies:
    """
    Container for vibrational bond energies.
    
    Attributes
    ----------
    gross : np.ndarray
        Gross vibrational bond energies (two-body only).
        For ZPEBOP-1, this equals net.
    net : np.ndarray
        Net vibrational bond energies.
        For ZPEBOP-1: same as gross.
        For ZPEBOP-2: gross + three-body contributions.
    composite : np.ndarray
        Combined matrix (gross upper triangle, net lower triangle).
    units : str
        Energy units ('kcal/mol').
    """
    gross: np.ndarray
    net: np.ndarray
    composite: np.ndarray
    units: str = 'kcal/mol'


@dataclass
class ZPEResult:
    """
    Results from a ZPEBOP calculation.
    
    Attributes
    ----------
    total_zpe : float
        Total zero-point vibrational energy.
    two_body : np.ndarray
        Two-body contributions matrix.
        For ZPEBOP-1: harmonic only.
        For ZPEBOP-2: harmonic + anharmonic.
    three_body_decomp : np.ndarray
        Three-body contributions decomposed to atom pairs.
        For ZPEBOP-1: zeros.
        For ZPEBOP-2: three-body coupling terms.
    atoms : np.ndarray
        Array of atomic symbols.
    model : str
        Model used ('zpebop1' or 'zpebop2').
    units : str
        Energy units ('kcal/mol').
    """
    total_zpe: float
    two_body: np.ndarray
    three_body_decomp: np.ndarray
    atoms: np.ndarray
    model: str
    units: str = 'kcal/mol'
    
    @property
    def n_atoms(self) -> int:
        """Number of atoms in the molecule."""
        return len(self.atoms)
    
    @property 
    def gross(self) -> np.ndarray:
        """Gross bond energies (two-body only)."""
        return self.two_body
    
    @property
    def net(self) -> np.ndarray:
        """Net bond energies (two-body + three-body)."""
        return self.two_body + self.three_body_decomp
    
    @property
    def two_body_sum(self) -> float:
        """Sum of two-body contributions."""
        return float(np.sum(self.two_body))
    
    @property
    def three_body_sum(self) -> float:
        """Sum of three-body contributions."""
        return float(np.sum(self.three_body_decomp))
