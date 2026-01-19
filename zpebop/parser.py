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
Optimized Gaussian output file parser for ZPEBOP.

This module provides high-performance functions to extract molecular data from 
Gaussian 16 output files generated with ROHF/CBSB3 and the Pop=(Full) keyword.

Performance optimizations:
    - Memory-mapped file reading for large files
    - Pre-allocated NumPy arrays
    - Minimized string operations

Required Gaussian Keywords
--------------------------
The output file must be generated with::

    # ROHF/CBSB3 Pop=(Full) IOp(6/27=122)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Union, Set
import mmap
import numpy as np

from .constants import SUPPORTED_ELEMENTS, ELEMENT_TO_INDEX

__all__ = ['MolecularData', 'parse_gaussian_output']

# Pre-compute supported elements as a set for O(1) lookup
_SUPPORTED_SET: Set[str] = set(SUPPORTED_ELEMENTS)


@dataclass
class MolecularData:
    """
    Container for molecular data extracted from Gaussian output.
    
    Attributes
    ----------
    atoms : np.ndarray
        1D array of element symbols (dtype=str) with shape (n_atoms,).
    coordinates : np.ndarray
        Cartesian coordinates in Angstroms with shape (n_atoms, 3).
    mulliken_bond_orders : np.ndarray
        Mulliken bond order matrix with shape (n_atoms, n_atoms).
    mulliken_charges : np.ndarray
        Mulliken atomic charges with shape (n_atoms,).
    atom_indices : np.ndarray
        Pre-computed element indices for vectorized operations.
    distance_matrix : np.ndarray
        Interatomic distance matrix with shape (n_atoms, n_atoms).
    """
    atoms: np.ndarray
    coordinates: np.ndarray
    mulliken_bond_orders: np.ndarray
    mulliken_charges: np.ndarray
    atom_indices: np.ndarray
    distance_matrix: np.ndarray
    
    @property
    def n_atoms(self) -> int:
        """Number of atoms in the molecule."""
        return len(self.atoms)


def compute_distance_matrix(coordinates: np.ndarray) -> np.ndarray:
    """
    Compute pairwise distance matrix from coordinates.
    
    Uses vectorized operations for efficiency.
    
    Parameters
    ----------
    coordinates : np.ndarray
        Cartesian coordinates with shape (n_atoms, 3).
    
    Returns
    -------
    np.ndarray
        Lower triangular distance matrix with shape (n_atoms, n_atoms).
    """
    n = len(coordinates)
    dist_matrix = np.zeros((n, n), dtype=np.float64)
    
    # Vectorized distance calculation for lower triangle
    for i in range(1, n):
        diff = coordinates[:i] - coordinates[i]
        dist_matrix[i, :i] = np.sqrt(np.sum(diff**2, axis=1))
    
    return dist_matrix


def _parse_atoms_fast(data: bytes, start_pos: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fast atom parsing using byte operations.
    
    Returns
    -------
    atoms : np.ndarray
        Array of element symbols
    atom_indices : np.ndarray
        Pre-computed indices into ELEMENT_TO_INDEX
    """
    atoms = []
    pos = start_pos
    
    # Find the end marker
    end_markers = (b'\n \n', b'\n       Variables:')
    end_pos = len(data)
    for marker in end_markers:
        idx = data.find(marker, pos)
        if idx != -1 and idx < end_pos:
            end_pos = idx
    
    while pos < end_pos:
        # Find end of line
        line_end = data.find(b'\n', pos)
        if line_end == -1 or line_end > end_pos:
            break
        
        # Extract first 3 characters and decode
        atom_bytes = data[pos:pos+3]
        try:
            atom = atom_bytes.decode('ascii').strip()
        except UnicodeDecodeError:
            pos = line_end + 1
            continue
        
        if atom in _SUPPORTED_SET:
            atoms.append(atom)
        
        pos = line_end + 1
    
    atoms_arr = np.array(atoms, dtype='U2')
    indices_arr = np.array([ELEMENT_TO_INDEX[a] for a in atoms], dtype=np.int32)
    
    return atoms_arr, indices_arr


def _parse_coordinates_fast(data: bytes, start_pos: int, n_atoms: int) -> np.ndarray:
    """Fast coordinate parsing with pre-allocated array."""
    coords = np.empty((n_atoms, 3), dtype=np.float64)
    pos = start_pos
    atom_idx = 0
    
    while atom_idx < n_atoms:
        line_end = data.find(b'\n', pos)
        if line_end == -1:
            break
        
        line = data[pos:line_end]
        
        # Check for end marker
        if line.strip().startswith(b'--------'):
            break
        
        # Parse coordinates from fixed positions (columns 35+)
        try:
            parts = line[34:].split()
            if len(parts) >= 3:
                coords[atom_idx, 0] = float(parts[0])
                coords[atom_idx, 1] = float(parts[1])
                coords[atom_idx, 2] = float(parts[2])
                atom_idx += 1
        except (ValueError, IndexError):
            pass
        
        pos = line_end + 1
    
    return coords[:atom_idx]


def _parse_mulliken_matrix_fast(data: bytes, start_pos: int, n_atoms: int) -> np.ndarray:
    """
    Fast Mulliken bond order matrix parsing.
    
    Handles Gaussian's block format (6 columns per block).
    """
    matrix = np.zeros((n_atoms, n_atoms), dtype=np.float64)
    pos = start_pos
    col_offset = 0
    row_in_block = 0
    
    # End marker
    end_marker = b'MBS Atomic-Atomic Spin Densities.'
    end_pos = data.find(end_marker, pos)
    if end_pos == -1:
        end_pos = len(data)
    
    while pos < end_pos:
        line_end = data.find(b'\n', pos)
        if line_end == -1 or line_end > end_pos:
            break
        
        line = data[pos:line_end]
        pos = line_end + 1
        
        # Skip empty lines
        stripped = line.strip()
        if not stripped:
            continue
        
        # Check if this is a header line (just numbers)
        try:
            parts = stripped.split()
            # Header lines have only integers
            if all(p.isdigit() or (p[0:1] == b'-' and p[1:].isdigit()) for p in parts):
                # Check if this is a new block
                first_num = int(parts[0])
                if first_num > 1 and row_in_block > 0:
                    col_offset += 6
                    row_in_block = 0
                continue
        except (ValueError, IndexError):
            pass
        
        # Parse data line: "  N  ELEM  val1  val2  ..."
        try:
            parts = line[12:].split()
            values = [float(v) for v in parts]
            
            if values and row_in_block < n_atoms:
                for j, val in enumerate(values):
                    col = col_offset + j
                    if col < n_atoms:
                        matrix[row_in_block, col] = val
                row_in_block += 1
        except (ValueError, IndexError):
            continue
    
    return matrix


def _parse_mulliken_charges_fast(data: bytes, start_pos: int, n_atoms: int) -> np.ndarray:
    """Fast Mulliken charges parsing."""
    charges = np.empty(n_atoms, dtype=np.float64)
    pos = start_pos
    atom_idx = 0
    
    end_marker = b'Sum of MBS Mulliken charges'
    end_pos = data.find(end_marker, pos)
    if end_pos == -1:
        end_pos = len(data)
    
    while pos < end_pos and atom_idx < n_atoms:
        line_end = data.find(b'\n', pos)
        if line_end == -1:
            break
        
        line = data[pos:line_end]
        try:
            # Charge is in columns 11-22
            charge_str = line[11:22].strip()
            charges[atom_idx] = float(charge_str)
            atom_idx += 1
        except (ValueError, IndexError):
            pass
        
        pos = line_end + 1
    
    return charges[:atom_idx]


def parse_gaussian_output(filepath: Union[str, Path]) -> MolecularData:
    """
    Parse a Gaussian output file and extract molecular data.
    
    Uses memory-mapped file reading for optimal performance with large files.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the Gaussian output file.
    
    Returns
    -------
    MolecularData
        Dataclass containing atoms, coordinates, bond orders, charges,
        and distance matrix.
    
    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    ValueError
        If required data sections are not found.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Gaussian output file not found: {filepath}")
    
    # Memory-map the file for efficient reading
    with open(filepath, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            data = bytes(mm)
    
    # Find key markers
    charge_marker = b' Charge ='
    coord_marker = b'Standard orientation:'
    mulliken_marker = b'MBS Condensed to atoms (all electrons):'
    charges_marker = b'MBS Mulliken charges and spin densities:'
    
    # Parse atoms
    charge_pos = data.find(charge_marker)
    if charge_pos == -1:
        raise ValueError("Could not find 'Charge =' in output file")
    
    # Find start of atom list (next line after Charge =)
    atom_start = data.find(b'\n', charge_pos) + 1
    atoms, atom_indices = _parse_atoms_fast(data, atom_start)
    n_atoms = len(atoms)
    
    if n_atoms == 0:
        raise ValueError("No atoms found in output file")
    
    # Parse coordinates (find last occurrence of Standard orientation)
    coord_pos = data.rfind(coord_marker)
    if coord_pos == -1:
        raise ValueError("Could not find 'Standard orientation' in output file")
    
    # Skip 5 lines to get to coordinate data
    pos = coord_pos
    for _ in range(5):
        pos = data.find(b'\n', pos) + 1
    
    coordinates = _parse_coordinates_fast(data, pos, n_atoms)
    
    # Compute distance matrix
    distance_matrix = compute_distance_matrix(coordinates)
    
    # Parse Mulliken bond orders
    mulliken_pos = data.find(mulliken_marker)
    if mulliken_pos == -1:
        raise ValueError("Could not find Mulliken bond orders in output file")
    
    # Skip 2 lines to get to data
    pos = mulliken_pos
    for _ in range(2):
        pos = data.find(b'\n', pos) + 1
    
    bond_orders = _parse_mulliken_matrix_fast(data, pos, n_atoms)
    
    # Parse Mulliken charges
    charges_pos = data.find(charges_marker)
    if charges_pos == -1:
        raise ValueError("Could not find Mulliken charges in output file")
    
    # Skip 2 lines
    pos = charges_pos
    for _ in range(2):
        pos = data.find(b'\n', pos) + 1
    
    charges = _parse_mulliken_charges_fast(data, pos, n_atoms)
    
    return MolecularData(
        atoms=atoms,
        coordinates=coordinates,
        mulliken_bond_orders=bond_orders,
        mulliken_charges=charges,
        atom_indices=atom_indices,
        distance_matrix=distance_matrix
    )
