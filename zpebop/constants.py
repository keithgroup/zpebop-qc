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
Physical constants and pre-computed parameter arrays for ZPEBOP.

This module contains parameters for both ZPEBOP-1 and ZPEBOP-2 models:

ZPEBOP-1 Parameters:
    - BETA_V1: Extended Hückel β parameters (18x18 array)

ZPEBOP-2 Parameters:
    - BETA_BOND, BETA_ANTI: Extended Hückel parameters (bonding/antibonding)
    - PRE_EXP, ZETA, R_PARAM: Short-range anharmonic parameters
    - KAPPA_BOND, KAPPA_ANTI: Three-body coupling parameters

All parameters are stored in 18x18 NumPy arrays indexed by element index,
enabling O(1) vectorized lookups with no file I/O.

References
----------
.. [1] Zulueta, B., Rude, C. D., Mangiardi, J. A., Petersson, G. A., & Keith, J. A.
       (2025). Zero-point energies from bond orders and populations relationships.
       The Journal of Chemical Physics, 162(8), 084102.
       https://doi.org/10.1063/5.0238831
"""

import numpy as np
from typing import Dict, Tuple

__all__ = [
    # Constants
    'HARTREE_TO_KCAL',
    'KCAL_TO_HARTREE',
    'SUPPORTED_ELEMENTS',
    'ELEMENT_TO_INDEX',
    'N_ELEMENTS',
    # ZPEBOP-1 parameters
    'BETA_V1_BOND',
    'BETA_V1_ANTI',
    # ZPEBOP-2 parameters
    'BETA_BOND',
    'BETA_ANTI',
    'PRE_EXP',
    'ZETA',
    'R_PARAM',
    'KAPPA_BOND',
    'KAPPA_ANTI',
    'HAS_KAPPA',
]

# =============================================================================
# Unit Conversion Factors
# =============================================================================

HARTREE_TO_KCAL: float = 627.5096
"""Conversion factor from Hartree to kcal/mol."""

KCAL_TO_HARTREE: float = 1.0 / HARTREE_TO_KCAL
"""Conversion factor from kcal/mol to Hartree."""

# =============================================================================
# Element Definitions
# =============================================================================

SUPPORTED_ELEMENTS: Tuple[str, ...] = (
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar'
)
"""Elements supported by ZPEBOP (H through Ar)."""

N_ELEMENTS: int = 18
"""Number of supported elements."""

ELEMENT_TO_INDEX: Dict[str, int] = {
    'H': 0, 'He': 1, 'Li': 2, 'Be': 3, 'B': 4, 'C': 5, 'N': 6, 'O': 7, 'F': 8,
    'Ne': 9, 'Na': 10, 'Mg': 11, 'Al': 12, 'Si': 13, 'P': 14, 'S': 15, 'Cl': 16, 'Ar': 17
}
"""Mapping from element symbol to array index (0-based)."""

nan = np.nan  # Shorthand for readability

# =============================================================================
# ZPEBOP-1 Parameters (18x18, indexed by element)
# Separate bonding and antibonding parameters
# Equation: E = 2 * β * P (bonding if P >= 0, antibonding if P < 0)
# Parameters are in kcal/mol (converted from Hartrees)
# =============================================================================

# Bonding parameters (used when bond order P >= 0)
BETA_V1_BOND = np.array([
    [7.887796, nan, 2.673191, 5.151854, 6.024092, 6.777104, 8.333327, 10.604912, 13.673434, nan, 4.969876, 5.515809, 5.572285, 6.055468, 7.354413, 8.101149, 8.433729, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [2.673191, nan, 0.702811, 0.834588, 1.073041, 1.587599, 1.361696, 0.646335, 3.281875, nan, 10.328808, 1.907629, 0.746736, 0.809487, 1.211094, 1.129517, 2.321786, nan],
    [5.151854, nan, 0.834588, 1.568774, 1.430722, 2.961845, 2.855169, 1.393071, 3.783883, nan, 4.681222, 1.713101, 1.411897, 1.223644, 1.474648, 1.581324, 2.748492, nan],
    [6.024092, nan, 1.073041, 1.430722, 2.371986, 2.497488, 1.656625, 0.878513, 4.059987, nan, 2.855169, 1.939005, 1.386796, 1.738202, 0.627510, 1.895079, 2.371986, nan],
    [6.777104, nan, 1.587599, 2.961845, 2.497488, 2.510038, 2.340611, 2.974396, 4.066262, nan, 1.706826, 5.352657, 2.604165, 1.731926, 0.119227, 2.089607, 2.371986, nan],
    [8.333327, nan, 1.361696, 2.855169, 1.656625, 2.340611, 2.516313, 4.505519, 5.710337, nan, 2.045681, 1.531123, 1.424447, 1.123242, 0.621235, 2.346886, 3.350901, nan],
    [10.604912, nan, 0.646335, 1.393071, 0.878513, 2.974396, 4.505519, 4.028612, 4.900850, nan, 0.131777, 1.236194, 1.939005, 1.688001, 1.945280, 2.171183, 1.876254, nan],
    [13.673434, nan, 3.281875, 3.783883, 4.059987, 4.066262, 5.710337, 4.900850, 5.992717, nan, 1.568774, 2.026856, 2.704566, 2.748492, 3.018321, 2.830068, 4.405117, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [4.969876, nan, 10.328808, 4.681222, 2.855169, 1.706826, 2.045681, 0.131777, 1.568774, nan, 1.449547, 1.167168, 1.474648, 1.110692, 0.928714, 0.734186, 1.223644, nan],
    [5.515809, nan, 1.907629, 1.713101, 1.939005, 5.352657, 1.531123, 1.236194, 2.026856, nan, 1.167168, 0.627510, 1.556224, 1.443272, 1.211094, 1.104417, 1.073041, nan],
    [5.572285, nan, 0.746736, 1.411897, 1.386796, 2.604165, 1.424447, 1.939005, 2.704566, nan, 1.474648, 1.556224, 1.223644, 1.066766, 1.179718, 1.506023, 2.039406, nan],
    [6.055468, nan, 0.809487, 1.223644, 1.738202, 1.731926, 1.123242, 1.688001, 2.748492, nan, 1.110692, 1.443272, 1.066766, 1.179718, 0.960090, 1.148343, 1.920179, nan],
    [7.354413, nan, 1.211094, 1.474648, 0.627510, 0.119227, 0.621235, 1.945280, 3.018321, nan, 0.928714, 1.211094, 1.179718, 0.960090, 1.430722, 1.255019, 1.913904, nan],
    [8.101149, nan, 1.129517, 1.581324, 1.895079, 2.089607, 2.346886, 2.171183, 2.830068, nan, 0.734186, 1.104417, 1.506023, 1.148343, 1.255019, 1.738202, 2.127258, nan],
    [8.433729, nan, 2.321786, 2.748492, 2.371986, 2.371986, 3.350901, 1.876254, 4.405117, nan, 1.223644, 1.073041, 2.039406, 1.920179, 1.913904, 2.127258, 2.208834, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
], dtype=np.float64)

# Antibonding parameters (used when bond order P < 0)
# These are positive values; the sign is handled in the equation
BETA_V1_ANTI = np.array([
    [7.887796, nan, 2.673191, 5.151854, 6.024092, 6.777104, 8.333327, 10.604912, 13.673434, nan, 4.969876, 5.515809, 5.572285, 6.055468, 7.354413, 8.101149, 8.433729, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [2.673191, nan, 0.702811, 0.834588, 1.073041, 1.587599, 1.361696, 0.646335, 3.281875, nan, 10.328808, 1.907629, 0.746736, 0.809487, 1.211094, 1.129517, 2.321786, nan],
    [5.151854, nan, 0.834588, 1.568774, 1.430722, 2.961845, 2.855169, 1.393071, 3.783883, nan, 4.681222, 1.713101, 1.411897, 1.223644, 1.474648, 1.581324, 2.748492, nan],
    [6.024092, nan, 1.073041, 1.430722, 2.371986, 2.497488, 1.656625, 0.878513, 4.059987, nan, 2.855169, 1.939005, 1.386796, 1.738202, 0.627510, 1.895079, 2.371986, nan],
    [6.777104, nan, 1.587599, 2.961845, 2.497488, 2.510038, 2.340611, 2.974396, 4.066262, nan, 1.706826, 5.352657, 2.604165, 1.731926, 0.119227, 2.089607, 2.371986, nan],
    [8.333327, nan, 1.361696, 2.855169, 1.656625, 2.340611, 2.516313, 4.505519, 5.710337, nan, 2.045681, 1.531123, 1.424447, 1.123242, 0.621235, 2.346886, 3.350901, nan],
    [10.604912, nan, 0.646335, 1.393071, 0.878513, 2.974396, 4.505519, 4.028612, 4.900850, nan, 0.131777, 1.236194, 1.939005, 1.688001, 1.945280, 2.171183, 1.876254, nan],
    [13.673434, nan, 3.281875, 3.783883, 4.059987, 4.066262, 5.710337, 4.900850, 5.992717, nan, 1.568774, 2.026856, 2.704566, 2.748492, 3.018321, 2.830068, 4.405117, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [4.969876, nan, 10.328808, 4.681222, 2.855169, 1.706826, 2.045681, 0.131777, 1.568774, nan, 1.449547, 1.167168, 1.474648, 1.110692, 0.928714, 0.734186, 1.223644, nan],
    [5.515809, nan, 1.907629, 1.713101, 1.939005, 5.352657, 1.531123, 1.236194, 2.026856, nan, 1.167168, 0.627510, 1.556224, 1.443272, 1.211094, 1.104417, 1.073041, nan],
    [5.572285, nan, 0.746736, 1.411897, 1.386796, 2.604165, 1.424447, 1.939005, 2.704566, nan, 1.474648, 1.556224, 1.223644, 1.066766, 1.179718, 1.506023, 2.039406, nan],
    [6.055468, nan, 0.809487, 1.223644, 1.738202, 1.731926, 1.123242, 1.688001, 2.748492, nan, 1.110692, 1.443272, 1.066766, 1.179718, 0.960090, 1.148343, 1.920179, nan],
    [7.354413, nan, 1.211094, 1.474648, 0.627510, 0.119227, 0.621235, 1.945280, 3.018321, nan, 0.928714, 1.211094, 1.179718, 0.960090, 1.430722, 1.255019, 1.913904, nan],
    [8.101149, nan, 1.129517, 1.581324, 1.895079, 2.089607, 2.346886, 2.171183, 2.830068, nan, 0.734186, 1.104417, 1.506023, 1.148343, 1.255019, 1.738202, 2.127258, nan],
    [8.433729, nan, 2.321786, 2.748492, 2.371986, 2.371986, 3.350901, 1.876254, 4.405117, nan, 1.223644, 1.073041, 2.039406, 1.920179, 1.913904, 2.127258, 2.208834, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
], dtype=np.float64)

# =============================================================================
# ZPEBOP-2 Parameters (18x18, indexed by element)
# Includes two-body harmonic, anharmonic short-range, and three-body terms
# =============================================================================

BETA_BOND = np.array([
    [7.66078979, nan, 4.30012927, 5.63016099, 5.76437285, 6.44929219, 7.86227855, 10.1303006, 13.0948875, nan, 3.6893398, 4.70422553, 5.25161116, 5.66618632, 6.77295588, 7.46816291, 7.81574821, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [4.30012927, nan, 0.702810879, 0.834588115, 1.07304073, 1.58759884, 1.3616963, 0.646335256, 3.28187512, nan, 10.3288078, 1.90762875, 0.74673604, 0.80948738, 1.21109391, 1.12951655, 2.32178623, nan],
    [5.63016099, nan, 0.834588115, 1.56877371, 1.43072242, 2.96184533, 2.85516859, 1.39307127, 3.78388266, nan, 4.68122196, 1.71310104, 1.41189703, 1.22364387, 1.47464841, 1.58132441, 2.74849197, nan],
    [5.76437285, nan, 1.07304073, 1.43072242, 2.62149668, 2.73088481, 2.04689927, 1.14188078, 3.7890867, nan, 2.85516859, 1.93900484, 1.38679627, 1.7382019, 0.627509786, 1.89507933, 2.37198637, nan],
    [6.44929219, nan, 1.58759884, 2.96184533, 2.73088481, 2.58295903, 2.38260298, 2.5976054, 3.57787817, nan, 1.70682576, 5.35265673, 2.60416483, 1.73192609, 0.119227253, 2.08960673, 2.37198637, nan],
    [7.86227855, nan, 1.3616963, 2.85516859, 2.04689927, 2.38260298, 2.7032261, 4.23440115, 4.9929025, nan, 2.04568131, 1.53112306, 1.42444718, 1.12324211, 0.62123529, 2.34688566, 3.35090141, nan],
    [10.1303006, nan, 0.646335256, 1.39307127, 1.14188078, 2.5976054, 4.23440115, 4.21595831, 4.66234813, nan, 0.131777345, 1.23619433, 1.93900484, 1.6880006, 1.94528046, 2.17118287, 1.87625361, nan],
    [13.0948875, nan, 3.28187512, 3.78388266, 3.7890867, 3.57787817, 4.9929025, 4.66234813, 5.99271693, nan, 1.56877371, 2.02685639, 2.7045663, 2.74849197, 3.01832128, 2.8300684, 4.40511656, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [3.6893398, nan, 10.3288078, 4.68122196, 2.85516859, 1.70682576, 2.04568131, 0.131777345, 1.56877371, nan, 1.44954697, 1.16716755, 1.47464841, 1.11069206, 0.928714357, 0.734186196, 1.22364387, nan],
    [4.70422553, nan, 1.90762875, 1.71310104, 1.93900484, 5.35265673, 1.53112306, 1.23619433, 2.02685639, nan, 1.16716755, 0.627509786, 1.55622397, 1.44327193, 1.21109391, 1.10441656, 1.07304073, nan],
    [5.25161116, nan, 0.74673604, 1.41189703, 1.38679627, 2.60416483, 1.42444718, 1.93900484, 2.7045663, nan, 1.47464841, 1.55622397, 1.22364387, 1.06676556, 1.17971792, 1.50602347, 2.03940568, nan],
    [5.66618632, nan, 0.80948738, 1.22364387, 1.7382019, 1.73192609, 1.12324211, 1.6880006, 2.74849197, nan, 1.11069206, 1.44327193, 1.06676556, 1.17971792, 0.960090393, 1.14834302, 1.92017933, nan],
    [6.77295588, nan, 1.21109391, 1.47464841, 0.627509786, 0.119227253, 0.62123529, 1.94528046, 3.01832128, nan, 0.928714357, 1.21109391, 1.17971792, 0.960090393, 1.43072242, 1.25501944, 1.91390384, nan],
    [7.46816291, nan, 1.12951655, 1.58132441, 1.89507933, 2.08960673, 2.34688566, 2.17118287, 2.8300684, nan, 0.734186196, 1.10441656, 1.50602347, 1.14834302, 1.25501944, 1.7382019, 2.12725789, nan],
    [7.81574821, nan, 2.32178623, 2.74849197, 2.37198637, 2.37198637, 3.35090141, 1.87625361, 4.40511656, nan, 1.22364387, 1.07304073, 2.03940568, 1.92017933, 1.91390384, 2.12725789, 2.20883368, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
], dtype=np.float64)

BETA_ANTI = np.array([
    [7.66078979, nan, 4.30012927, 5.63016099, 5.76437285, 6.44929219, 7.86227855, 10.1303006, 13.0948875, nan, 3.6893398, 4.70422553, 5.25161116, 5.66618632, 6.77295588, 7.46816291, 7.81574821, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [4.30012927, nan, 0.702810879, 0.834588115, 1.07304073, 1.58759884, 1.3616963, 0.646335256, 3.28187512, nan, 10.3288078, 1.90762875, 0.74673604, 0.80948738, 1.21109391, 1.12951655, 2.32178623, nan],
    [5.63016099, nan, 0.834588115, 1.56877371, 1.43072242, 2.96184533, 2.85516859, 1.39307127, 3.78388266, nan, 4.68122196, 1.71310104, 1.41189703, 1.22364387, 1.47464841, 1.58132441, 2.74849197, nan],
    [5.76437285, nan, 1.07304073, 1.43072242, 2.62149668, 2.73088481, 2.04689927, 1.14188078, 3.7890867, nan, 2.85516859, 1.93900484, 1.38679627, 1.7382019, 0.627509786, 1.89507933, 2.37198637, nan],
    [6.44929219, nan, 1.58759884, 2.96184533, 2.73088481, 2.58295903, 2.38260298, 2.5976054, 3.57787817, nan, 1.70682576, 5.35265673, 2.60416483, 1.73192609, 0.119227253, 2.08960673, 2.37198637, nan],
    [7.86227855, nan, 1.3616963, 2.85516859, 2.04689927, 2.38260298, 2.7032261, 4.23440115, 4.9929025, nan, 2.04568131, 1.53112306, 1.42444718, 1.12324211, 0.62123529, 2.34688566, 3.35090141, nan],
    [10.1303006, nan, 0.646335256, 1.39307127, 1.14188078, 2.5976054, 4.23440115, 4.21595831, 4.66234813, nan, 0.131777345, 1.23619433, 1.93900484, 1.6880006, 1.94528046, 2.17118287, 1.87625361, nan],
    [13.0948875, nan, 3.28187512, 3.78388266, 3.7890867, 3.57787817, 4.9929025, 4.66234813, 5.99271693, nan, 1.56877371, 2.02685639, 2.7045663, 2.74849197, 3.01832128, 2.8300684, 4.40511656, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [3.6893398, nan, 10.3288078, 4.68122196, 2.85516859, 1.70682576, 2.04568131, 0.131777345, 1.56877371, nan, 1.44954697, 1.16716755, 1.47464841, 1.11069206, 0.928714357, 0.734186196, 1.22364387, nan],
    [4.70422553, nan, 1.90762875, 1.71310104, 1.93900484, 5.35265673, 1.53112306, 1.23619433, 2.02685639, nan, 1.16716755, 0.627509786, 1.55622397, 1.44327193, 1.21109391, 1.10441656, 1.07304073, nan],
    [5.25161116, nan, 0.74673604, 1.41189703, 1.38679627, 2.60416483, 1.42444718, 1.93900484, 2.7045663, nan, 1.47464841, 1.55622397, 1.22364387, 1.06676556, 1.17971792, 1.50602347, 2.03940568, nan],
    [5.66618632, nan, 0.80948738, 1.22364387, 1.7382019, 1.73192609, 1.12324211, 1.6880006, 2.74849197, nan, 1.11069206, 1.44327193, 1.06676556, 1.17971792, 0.960090393, 1.14834302, 1.92017933, nan],
    [6.77295588, nan, 1.21109391, 1.47464841, 0.627509786, 0.119227253, 0.62123529, 1.94528046, 3.01832128, nan, 0.928714357, 1.21109391, 1.17971792, 0.960090393, 1.43072242, 1.25501944, 1.91390384, nan],
    [7.46816291, nan, 1.12951655, 1.58132441, 1.89507933, 2.08960673, 2.34688566, 2.17118287, 2.8300684, nan, 0.734186196, 1.10441656, 1.50602347, 1.14834302, 1.25501944, 1.7382019, 2.12725789, nan],
    [7.81574821, nan, 2.32178623, 2.74849197, 2.37198637, 2.37198637, 3.35090141, 1.87625361, 4.40511656, nan, 1.22364387, 1.07304073, 2.03940568, 1.92017933, 1.91390384, 2.12725789, 2.20883368, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
], dtype=np.float64)

# Pre-exponential factors for short-range anharmonic correction
PRE_EXP = np.array([
    [1.68693089099376, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, 5.8893251685497, 7.21568339508251, 6.23858631795688, 1.6693426234428, 5.46285076195918, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, 7.21568339508251, 8.08422424093555, 6.74949406261867, 5.02133181682595, 5.93831968025992, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, 6.23858631795688, 6.74949406261867, 5.7704259823636, 2.17015568295891, 5.99973823102167, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, 1.6693426234428, 5.02133181682595, 2.17015568295891, 7.87098093628823, 4.21960952037527, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, 5.46285076195918, 5.93831968025992, 5.99973823102167, 4.21960952037527, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
], dtype=np.float64)

# Zeta (exponent) parameters for short-range anharmonic correction
ZETA = np.array([
    [2.41912455668393, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, 1.92312714133611, 2.15089303685116, 2.39117174556478, 1.28704839759048, 1.91759851200377, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, 2.15089303685116, 2.37252376839168, 2.64217437149461, 2.04605152389389, 2.06618019619969, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, 2.39117174556478, 2.64217437149461, 2.11587695606116, 0.933135509854831, 2.14063963656117, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, 1.28704839759048, 2.04605152389389, 0.933135509854831, 2.61181009686831, 1.72787656847102, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, 1.91759851200377, 2.06618019619969, 2.14063963656117, 1.72787656847102, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
], dtype=np.float64)

# R (distance) parameters for short-range anharmonic correction
R_PARAM = np.array([
    [0.833682918626989, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, 1.55376377968915, 1.54232618109695, 1.52315247688686, 1.23755181574759, 1.42102988765131, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, 1.54232618109695, 1.52693139866426, 1.44605393633299, 1.25645968851403, 1.3696809413131, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, 1.52315247688686, 1.44605393633299, 1.20497012992851, 1.08095116549028, 1.30621232251096, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, 1.23755181574759, 1.25645968851403, 1.08095116549028, 1.1932085823802, 1.17498665398655, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, 1.42102988765131, 1.3696809413131, 1.30621232251096, 1.17498665398655, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
], dtype=np.float64)

# Three-body kappa bonding parameters
KAPPA_BOND = np.array([
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, 6.35971211545447, 5.46097259289613, 4.61514758141805, 1.12787484479824, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, 5.46097259289613, 4.25893505849368, 4.22125686188039, 2.7224736105841, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, 4.61514758141805, 4.22125686188039, 2.6770795564117, 11.0556993804226, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, 1.12787484479824, 2.7224736105841, 11.0556993804226, 73.3391927480235, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
], dtype=np.float64)

# Three-body kappa antibonding parameters
KAPPA_ANTI = np.array([
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, 2.3561185829866e-13, 24.8167973203653, 8.37845322326246, 97.8067916424023, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, 24.8167973203653, 0.791597061589119, 2.47782495134405, 3.35702346413509e-12, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, 8.37845322326246, 2.47782495134405, 11.6737472358475, 1.33652210042449e-14, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, 97.8067916424023, 3.35702346413509e-12, 1.33652210042449e-14, 0.088504273757526, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
], dtype=np.float64)

# Make all arrays read-only
for _arr in [BETA_V1_BOND, BETA_V1_ANTI, BETA_BOND, BETA_ANTI, PRE_EXP, ZETA, R_PARAM, KAPPA_BOND, KAPPA_ANTI]:
    _arr.flags.writeable = False

# Pre-computed boolean mask for three-body terms
HAS_KAPPA = ~np.isnan(KAPPA_BOND)
HAS_KAPPA.flags.writeable = False
