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
Core ZPEBOP calculator with unified interface for all models.

This module provides the main ZPECalculator class that supports
both ZPEBOP-1 and ZPEBOP-2 models through a single interface.

Examples
--------
>>> from zpebop import ZPECalculator
>>> 
>>> # Use ZPEBOP-2 (default)
>>> calc = ZPECalculator("molecule.out")
>>> result = calc.compute_zpe()
>>> 
>>> # Use ZPEBOP-1
>>> calc = ZPECalculator("molecule.out", model="zpebop1")
>>> result = calc.compute_zpe()
>>>
>>> # With isotope correction (deuterium at atom 1)
>>> result_iso = calc.compute_zpe_isotope({1: 2.014102})
"""

from pathlib import Path
from typing import Tuple, Union, Dict, Optional
import numpy as np

from .constants import ELEMENT_TO_INDEX, ATOMIC_MASSES
from .parser import MolecularData, parse_gaussian_output
from .models.base import ZPEResult, BondEnergies, IsotopeZPEResult
from .models.zpebop1 import compute_zpe_v1
from .models.zpebop2 import compute_zpe_v2

__all__ = ['ZPECalculator']

# Valid model names
VALID_MODELS = {'zpebop1', 'zpebop2'}


class ZPECalculator:
    """
    Unified ZPEBOP calculator for zero-point vibrational energies.
    
    Supports both ZPEBOP-1 (harmonic only) and ZPEBOP-2 (harmonic + 
    anharmonic + three-body) models through a single interface.
    
    Parameters
    ----------
    source : str, Path, or MolecularData
        Path to Gaussian output file or MolecularData object.
    model : str, optional
        ZPEBOP model to use: 'zpebop1' or 'zpebop2' (default).
    
    Attributes
    ----------
    model : str
        The selected ZPEBOP model.
    atoms : np.ndarray
        Array of atomic symbols.
    n_atoms : int
        Number of atoms in the molecule.
    
    Examples
    --------
    >>> calc = ZPECalculator("molecule.out")
    >>> result = calc.compute_zpe()
    >>> print(f"ZPE = {result.total_zpe:.3f} kcal/mol")
    
    >>> calc = ZPECalculator("molecule.out", model="zpebop1")
    >>> result = calc.compute_zpe()
    >>> print(f"ZPE (harmonic) = {result.total_zpe:.3f} kcal/mol")
    """
    
    def __init__(self, source: Union[str, Path, MolecularData], 
                 model: str = 'zpebop2'):
        # Validate model
        if model not in VALID_MODELS:
            raise ValueError(f"Invalid model '{model}'. Must be one of: {VALID_MODELS}")
        
        self.model = model
        
        # Parse input
        if isinstance(source, MolecularData):
            self.mol_data = source
        else:
            self.mol_data = parse_gaussian_output(source)
        
        self.atoms = self.mol_data.atoms
        self.atom_indices = self.mol_data.atom_indices
        self.bond_orders = self.mol_data.mulliken_bond_orders
        self.distance_matrix = self.mol_data.distance_matrix
        
        # Cache results
        self._result = None
    
    @property
    def n_atoms(self) -> int:
        """Number of atoms in the molecule."""
        return len(self.atoms)
    
    def compute_zpe(self) -> ZPEResult:
        """
        Compute the zero-point vibrational energy.
        
        Returns
        -------
        ZPEResult
            Result object containing total ZPE and bond contributions.
        """
        if self._result is not None:
            return self._result
        
        if self.model == 'zpebop1':
            total_zpe, two_body, three_body_decomp = compute_zpe_v1(
                self.atom_indices, self.bond_orders
            )
        else:  # zpebop2
            total_zpe, two_body, three_body_decomp = compute_zpe_v2(
                self.atom_indices, self.bond_orders, self.distance_matrix
            )
        
        self._result = ZPEResult(
            total_zpe=total_zpe,
            two_body=two_body,
            three_body_decomp=three_body_decomp,
            atoms=self.atoms,
            model=self.model,
            units='kcal/mol'
        )
        
        return self._result
    
    def compute_bond_energies(self) -> BondEnergies:
        """
        Compute vibrational bond energy tables.
        
        Returns
        -------
        BondEnergies
            Dataclass containing gross, net, and composite matrices.
            For ZPEBOP-1, gross and net are identical.
        """
        result = self.compute_zpe()
        
        n = self.n_atoms
        gross = result.two_body
        net = result.two_body + result.three_body_decomp
        
        # Build composite table
        composite = np.zeros((n, n), dtype=np.float64)
        for i in range(1, n):
            for j in range(i):
                composite[j, i] = gross[i, j]  # Upper: gross
                composite[i, j] = net[i, j]    # Lower: net
        
        return BondEnergies(gross=gross, net=net, composite=composite, 
                          units='kcal/mol')
    
    def sort_bond_energies(self, bond_energy_matrix: np.ndarray,
                           relative: bool = False,
                           include_atom_numbers: bool = True,
                           threshold: float = 0.01,
                           isotopes: Dict[int, float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sort bond energies from weakest to strongest.
        
        Parameters
        ----------
        bond_energy_matrix : np.ndarray
            Bond energy matrix from compute_bond_energies().
        relative : bool, optional
            If True, return energies relative to minimum.
        include_atom_numbers : bool, optional
            If True, include atom indices in labels.
        threshold : float, optional
            Minimum energy to include.
        isotopes : dict, optional
            Mapping of atom number (1-indexed) to isotope mass.
            Atoms with isotope substitutions will be marked with *.
        
        Returns
        -------
        bond_labels : np.ndarray
            Bond identity strings.
        bond_energies : np.ndarray
            Sorted bond energies.
        """
        n = self.n_atoms
        labels = []
        energies = []
        
        for i in range(1, n):
            for j in range(i):
                e = bond_energy_matrix[i, j]
                if e < threshold:
                    continue
                
                if include_atom_numbers:
                    # Add asterisk for isotope-substituted atoms
                    marker_j = '*' if isotopes and (j + 1) in isotopes else ''
                    marker_i = '*' if isotopes and (i + 1) in isotopes else ''
                    label = f"{self.atoms[j]}{j+1}{marker_j}-{self.atoms[i]}{i+1}{marker_i}"
                else:
                    label = f"{self.atoms[j]}-{self.atoms[i]}"
                
                labels.append(label)
                energies.append(e)
        
        # Sort by energy
        labels = np.array(labels)
        energies = np.array(energies)
        sort_idx = np.argsort(energies)
        
        labels = labels[sort_idx]
        energies = energies[sort_idx]
        
        if relative and len(energies) > 0:
            energies = energies - energies[0]
        
        return labels, energies
    
    def _compute_isotope_correction_matrix(self, 
                                            isotopes: Dict[int, float]) -> np.ndarray:
        """
        Compute matrix of isotope correction factors sqrt(μ_1/μ_2).
        
        Parameters
        ----------
        isotopes : dict
            Mapping of atom number (1-indexed) to isotope mass.
        
        Returns
        -------
        np.ndarray
            Matrix of correction factors (lower triangular).
        """
        n = self.n_atoms
        correction = np.ones((n, n), dtype=np.float64)
        
        for i in range(1, n):
            for j in range(i):
                # Standard masses
                m1_i = ATOMIC_MASSES[self.atoms[i]]
                m1_j = ATOMIC_MASSES[self.atoms[j]]
                mu_1 = (m1_i * m1_j) / (m1_i + m1_j)
                
                # Isotope masses (use standard if not specified)
                # Note: isotopes dict uses 1-indexed atom numbers
                m2_i = isotopes.get(i + 1, m1_i)
                m2_j = isotopes.get(j + 1, m1_j)
                mu_2 = (m2_i * m2_j) / (m2_i + m2_j)
                
                # Correction factor: sqrt(μ_1/μ_2)
                correction[i, j] = np.sqrt(mu_1 / mu_2)
        
        return correction
    
    def compute_zpe_isotope(self, isotopes: Dict[int, float]) -> IsotopeZPEResult:
        """
        Compute ZPE with isotope mass corrections.
        
        Uses the harmonic oscillator approximation:
            BE_isotope = BE_normal * sqrt(μ_normal / μ_isotope)
        
        where μ = m₁*m₂/(m₁+m₂) is the reduced mass.
        
        Parameters
        ----------
        isotopes : dict
            Mapping of atom number (1-indexed) to isotope mass.
            Example: {1: 2.014102} for deuterium at atom 1.
            
            Common isotope masses:
            - Deuterium (D): 2.014102
            - Tritium (T): 3.016049
            - Carbon-13: 13.00335
            - Carbon-14: 14.00324
            - Nitrogen-15: 15.00011
            - Oxygen-18: 17.99916
        
        Returns
        -------
        IsotopeZPEResult
            Result with isotope-corrected energies and comparison data.
        
        Examples
        --------
        >>> calc = ZPECalculator("molecule.out")
        >>> # Replace H at atom 1 with deuterium
        >>> result = calc.compute_zpe_isotope({1: 2.014102})
        >>> print(f"Normal ZPE: {result.total_zpe_normal:.3f}")
        >>> print(f"Isotope ZPE: {result.total_zpe:.3f}")
        >>> print(f"ΔZPE: {result.zpe_difference:.3f}")
        """
        # Validate isotope atom numbers
        for atom_num in isotopes:
            if atom_num < 1 or atom_num > self.n_atoms:
                raise ValueError(
                    f"Invalid atom number {atom_num}. "
                    f"Must be between 1 and {self.n_atoms}."
                )
        
        # Compute normal ZPE first
        normal_result = self.compute_zpe()
        
        # Compute correction factors
        correction = self._compute_isotope_correction_matrix(isotopes)
        
        # Apply corrections to two-body terms
        two_body_isotope = normal_result.two_body * correction
        
        # Apply corrections to three-body decomposition
        three_body_isotope = normal_result.three_body_decomp * correction
        
        # Total isotope-corrected ZPE
        total_zpe_isotope = np.sum(two_body_isotope) + np.sum(three_body_isotope)
        
        return IsotopeZPEResult(
            total_zpe=total_zpe_isotope,
            total_zpe_normal=normal_result.total_zpe,
            two_body=two_body_isotope,
            two_body_normal=normal_result.two_body,
            three_body_decomp=three_body_isotope,
            three_body_decomp_normal=normal_result.three_body_decomp,
            correction_factors=correction,
            atoms=self.atoms,
            isotopes=isotopes,
            model=self.model,
            units='kcal/mol'
        )
    
    def compute_bond_energies_isotope(self, 
                                       isotopes: Dict[int, float]) -> Tuple[BondEnergies, BondEnergies]:
        """
        Compute vibrational bond energy tables with isotope corrections.
        
        Parameters
        ----------
        isotopes : dict
            Mapping of atom number (1-indexed) to isotope mass.
        
        Returns
        -------
        normal_energies : BondEnergies
            Bond energies without isotope correction.
        isotope_energies : BondEnergies
            Bond energies with isotope correction.
        """
        result = self.compute_zpe_isotope(isotopes)
        
        n = self.n_atoms
        
        # Normal bond energies
        gross_normal = result.two_body_normal
        net_normal = result.two_body_normal + result.three_body_decomp_normal
        composite_normal = np.zeros((n, n), dtype=np.float64)
        for i in range(1, n):
            for j in range(i):
                composite_normal[j, i] = gross_normal[i, j]
                composite_normal[i, j] = net_normal[i, j]
        
        normal = BondEnergies(
            gross=gross_normal, 
            net=net_normal, 
            composite=composite_normal
        )
        
        # Isotope-corrected bond energies
        gross_iso = result.two_body
        net_iso = result.two_body + result.three_body_decomp
        composite_iso = np.zeros((n, n), dtype=np.float64)
        for i in range(1, n):
            for j in range(i):
                composite_iso[j, i] = gross_iso[i, j]
                composite_iso[i, j] = net_iso[i, j]
        
        isotope = BondEnergies(
            gross=gross_iso, 
            net=net_iso, 
            composite=composite_iso
        )
        
        return normal, isotope
