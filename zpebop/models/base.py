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

from dataclasses import dataclass, field
from typing import Tuple, Dict, Optional
import numpy as np

__all__ = ['ZPEResult', 'BondEnergies', 'IsotopeZPEResult']


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


@dataclass
class IsotopeZPEResult:
    """
    Results from a ZPEBOP calculation with isotope corrections.
    
    The isotope correction uses the harmonic oscillator approximation:
        BE_isotope = BE_normal * sqrt(μ_normal / μ_isotope)
    
    where μ = m₁*m₂/(m₁+m₂) is the reduced mass of the bond.
    
    Attributes
    ----------
    total_zpe : float
        Total isotope-corrected zero-point vibrational energy.
    total_zpe_normal : float
        Total ZPE without isotope correction (for comparison).
    two_body : np.ndarray
        Isotope-corrected two-body contributions matrix.
    two_body_normal : np.ndarray
        Original two-body contributions (no isotope correction).
    three_body_decomp : np.ndarray
        Isotope-corrected three-body contributions.
    three_body_decomp_normal : np.ndarray
        Original three-body contributions.
    correction_factors : np.ndarray
        Matrix of sqrt(μ_normal/μ_isotope) correction factors.
    atoms : np.ndarray
        Array of atomic symbols.
    isotopes : Dict[int, float]
        Mapping of atom number (1-indexed) to isotope mass.
    model : str
        Model used ('zpebop1' or 'zpebop2').
    units : str
        Energy units ('kcal/mol').
    """
    total_zpe: float
    total_zpe_normal: float
    two_body: np.ndarray
    two_body_normal: np.ndarray
    three_body_decomp: np.ndarray
    three_body_decomp_normal: np.ndarray
    correction_factors: np.ndarray
    atoms: np.ndarray
    isotopes: Dict[int, float]
    model: str
    units: str = 'kcal/mol'
    
    @property
    def n_atoms(self) -> int:
        """Number of atoms in the molecule."""
        return len(self.atoms)
    
    @property
    def gross(self) -> np.ndarray:
        """Isotope-corrected gross bond energies (two-body only)."""
        return self.two_body
    
    @property
    def gross_normal(self) -> np.ndarray:
        """Original gross bond energies (no isotope correction)."""
        return self.two_body_normal
    
    @property
    def net(self) -> np.ndarray:
        """Isotope-corrected net bond energies (two-body + three-body)."""
        return self.two_body + self.three_body_decomp
    
    @property
    def net_normal(self) -> np.ndarray:
        """Original net bond energies (no isotope correction)."""
        return self.two_body_normal + self.three_body_decomp_normal
    
    @property
    def zpe_ratio(self) -> float:
        """Ratio of isotope ZPE to normal ZPE."""
        if self.total_zpe_normal > 0:
            return self.total_zpe / self.total_zpe_normal
        return 1.0
    
    @property
    def zpe_difference(self) -> float:
        """Difference between normal and isotope ZPE (normal - isotope)."""
        return self.total_zpe_normal - self.total_zpe
    
    def get_isotope_label(self, atom_num: int) -> str:
        """
        Get a label for an atom showing isotope mass if substituted.
        
        Parameters
        ----------
        atom_num : int
            Atom number (1-indexed).
        
        Returns
        -------
        str
            Label like "C1" or "D1" or "C1(13.003)"
        """
        idx = atom_num - 1
        symbol = self.atoms[idx]
        
        if atom_num in self.isotopes:
            mass = self.isotopes[atom_num]
            # Check for common isotope names
            if symbol == 'H':
                if abs(mass - 2.014) < 0.01:
                    return f"D{atom_num}"
                elif abs(mass - 3.016) < 0.01:
                    return f"T{atom_num}"
            return f"{symbol}{atom_num}({mass:.3f})"
        return f"{symbol}{atom_num}"
