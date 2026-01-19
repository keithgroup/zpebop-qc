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
ZPEBOP: Zero-Point Energies from Bond Orders and Populations.

A Python package for computing molecular zero-point vibrational energies
from Mulliken bond orders obtained from ROHF/CBSB3 calculations.

Available Models
----------------
- **ZPEBOP-1**: Harmonic term only (E = 2 * β * |P|)
- **ZPEBOP-2**: Harmonic + anharmonic + three-body coupling terms

Quick Start
-----------
>>> from zpebop import ZPECalculator
>>> 
>>> # Use ZPEBOP-2 (default, most accurate)
>>> calc = ZPECalculator("molecule.out")
>>> result = calc.compute_zpe()
>>> print(f"ZPE = {result.total_zpe:.3f} kcal/mol")
>>> 
>>> # Use ZPEBOP-1 (harmonic only)
>>> calc = ZPECalculator("molecule.out", model="zpebop1")
>>> result = calc.compute_zpe()
>>> print(f"ZPE = {result.total_zpe:.3f} kcal/mol")

Command Line
------------
::

    zpebop -f molecule.out                    # Default: ZPEBOP-2
    zpebop -f molecule.out --model zpebop1    # Use ZPEBOP-1
    zpebop -f molecule.out --be --sort        # With bond energies

References
----------
.. [1] Zulueta, B., Rude, C. D., Mangiardi, J. A., Petersson, G. A., & Keith, J. A.
       (2025). Zero-point energies from bond orders and populations relationships.
       The Journal of Chemical Physics, 162(8), 084102.
       https://doi.org/10.1063/5.0238831
"""

__version__ = '1.0.0'
__author__ = 'Barbaro Zulueta'
__email__ = 'blz11@pitt.edu'

# Core calculator
from .core import ZPECalculator

# Data classes
from .models.base import ZPEResult, BondEnergies, IsotopeZPEResult

# Parser
from .parser import MolecularData, parse_gaussian_output

# Constants
from .constants import (
    HARTREE_TO_KCAL,
    KCAL_TO_HARTREE,
    SUPPORTED_ELEMENTS,
    ELEMENT_TO_INDEX,
    N_ELEMENTS,
    ATOMIC_MASSES,
    COMMON_ISOTOPES,
)

# Public API
__all__ = [
    # Version info
    '__version__',
    '__author__',
    
    # Core
    'ZPECalculator',
    
    # Data classes
    'ZPEResult',
    'BondEnergies',
    'IsotopeZPEResult',
    
    # Parser
    'MolecularData',
    'parse_gaussian_output',
    
    # Constants
    'HARTREE_TO_KCAL',
    'KCAL_TO_HARTREE',
    'SUPPORTED_ELEMENTS',
    'ELEMENT_TO_INDEX',
    'N_ELEMENTS',
    'ATOMIC_MASSES',
    'COMMON_ISOTOPES',
]


def get_version() -> str:
    """Return the package version string."""
    return __version__


def cite() -> str:
    """
    Return citation information for ZPEBOP.
    
    Returns
    -------
    str
        Citation string for publications.
    """
    return (
        "ZPEBOP: Zero-Point Energies from Bond Orders and Populations\n"
        "Version: {}\n"
        "Author: Barbaro Zulueta\n"
        "University of Pittsburgh\n\n"
        "If you use ZPEBOP in your research, please cite:\n"
        "Zulueta, B., Rude, C. D., Mangiardi, J. A., Petersson, G. A., & Keith, J. A.\n"
        "(2025). Zero-point energies from bond orders and populations relationships.\n"
        "The Journal of Chemical Physics, 162(8), 084102.\n"
        "https://doi.org/10.1063/5.0238831"
    ).format(__version__)
