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
"""

from pathlib import Path
from typing import Tuple, Union
import numpy as np

from .constants import ELEMENT_TO_INDEX
from .parser import MolecularData, parse_gaussian_output
from .models.base import ZPEResult, BondEnergies
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
                           threshold: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
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
                    label = f"{self.atoms[j]}{j+1}-{self.atoms[i]}{i+1}"
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
