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
from typing import Dict, Tuple, Optional

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
    # Atomic masses
    'ATOMIC_MASSES',
    'COMMON_ISOTOPES',
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
    [26.267552, nan, 4.681222, 12.506266, 25.602392, 20.588590, 28.131255, 52.779832, 34.607154, nan, 0.000000, 2.510038, 27.773575, 23.086078, 45.701524, 31.457056, 24.284622, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [4.681222, nan, 4.216865, 2.572789, 5.704062, 0.000000, 0.000000, 8.847885, nan, nan, nan, nan, nan, 19.885779, nan, nan, nan, nan],
    [12.506266, nan, 2.572789, 1.386796, 3.407377, 0.000000, 0.000000, 0.000000, 5.478159, nan, nan, nan, nan, 7.881521, 0.571034, 0.000000, 0.150602, nan],
    [25.602392, nan, 5.704062, 3.407377, 10.297433, 0.000000, -21.887535, 39.708807, 11.188496, nan, nan, 0.307480, nan, -0.527108, 0.000000, 9.236941, 2.528864, nan],
    [20.588590, nan, 0.000000, 0.000000, 0.000000, 4.229415, 6.036642, 7.034383, 11.703054, nan, 0.000000, 0.000000, 0.000000, 6.168419, 1.255019, 8.107424, 4.373742, nan],
    [28.131255, nan, 0.000000, 0.000000, -21.887535, 6.036642, 20.488188, 5.258530, 13.554207, nan, nan, 1.556224, 0.000000, 15.838342, 46.059205, 23.625736, 0.000000, nan],
    [52.779832, nan, 8.847885, 0.000000, 39.708807, 7.034383, 5.258530, 15.794417, 22.653097, nan, nan, nan, nan, 0.000000, 10.240957, 13.384780, 12.958073, nan],
    [34.607154, nan, nan, 5.478159, 11.188496, 11.703054, 13.554207, 22.653097, 16.296424, nan, nan, nan, nan, 7.856420, 0.000000, 15.279859, 1.650350, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [0.000000, nan, nan, nan, nan, 0.000000, nan, nan, nan, nan, 0.000000, 3.357176, nan, nan, nan, nan, nan, nan],
    [2.510038, nan, nan, nan, 0.307480, 0.000000, 1.556224, nan, nan, nan, 3.357176, nan, nan, nan, 0.614959, nan, nan, nan],
    [27.773575, nan, nan, nan, nan, 0.000000, 0.000000, nan, nan, nan, nan, nan, 0.689006, nan, nan, 0.000000, nan, nan],
    [23.086078, nan, 19.885779, 7.881521, -0.527108, 6.168419, 15.838342, 0.000000, 7.856420, nan, nan, nan, nan, 0.533383, 6.237445, 6.933981, 14.853152, nan],
    [45.701524, nan, nan, 0.571034, 0.000000, 1.255019, 46.059205, 10.240957, 0.000000, nan, nan, 0.614959, nan, 6.237445, 21.128248, 11.113195, 5.678962, nan],
    [31.457056, nan, nan, 0.000000, 9.236941, 8.107424, 23.625736, 13.384780, 15.279859, nan, nan, nan, 0.000000, 6.933981, 11.113195, 12.782371, 0.000000, nan],
    [24.284622, nan, nan, 0.150602, 2.528864, 4.373742, 0.000000, 12.958073, 1.650350, nan, nan, nan, nan, 14.853152, 5.678962, 0.000000, 3.081072, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
], dtype=np.float64)

# =============================================================================
# ZPEBOP-2 Parameters (18x18, indexed by element)
# Harmonic + Anharmonic + Three-body terms
# =============================================================================

BETA_BOND = np.array([
    [7.85752443462511, nan, 2.64643153802927, 4.99851472511061, 11.562020311103, 6.63831874335757, 8.15201199094122, 10.7892439321813, 13.4601465149703, nan, 4.5347132160903, nan, nan, nan, nan, nan, 7.93033870668636, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [2.64643153802927, nan, 0.690489207627185, 1.82521188836532, 2.17858304224393, 5.13848977491812, 13.2531166741975, 3.16566446811649, 3.69641601424705, nan, 7.34448369614239, nan, nan, nan, nan, nan, 2.35730963561654, nan],
    [4.99851472511061, nan, 1.82521188836532, 1.61638200421122, 2.28356516663285, 1.34918691374378, 4.32691669180403, 52.1623490407304, 15.4178757435101, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [11.562020311103, nan, 2.17858304224393, 2.28356516663285, 18.9861037711278, 13.1323521699709, 5.58908111732033, 19.6512402264327, 7.06216684114365, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [6.63831874335757, nan, 5.13848977491812, 1.34918691374378, 13.1323521699709, 3.43251054167047, 8.05691389667233, 4.44217100714648, 5.98485647124534, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [8.15201199094122, nan, 13.2531166741975, 4.32691669180403, 5.58908111732033, 8.05691389667233, 26.9637243707608, 10.0338957043055, 13.9613810965585, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [10.7892439321813, nan, 3.16566446811649, 52.1623490407304, 19.6512402264327, 4.44217100714648, 10.0338957043055, 11.8580299536351, 5.82727820591931, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [13.4601465149703, nan, 3.69641601424705, 15.4178757435101, 7.06216684114365, 5.98485647124534, 13.9613810965585, 5.82727820591931, 5.91325290786045, nan, 1.67691795591411, nan, nan, nan, nan, nan, 6.49273582515203, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [4.5347132160903, nan, 7.34448369614239, nan, nan, nan, nan, nan, 1.67691795591411, nan, 0.601726456223389, nan, nan, nan, nan, nan, 1.21486964468664, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [7.93033870668636, nan, 2.35730963561654, nan, nan, nan, nan, nan, 6.49273582515203, nan, 1.21486964468664, nan, nan, nan, nan, nan, 1.94140588157533, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
], dtype=np.float64)

BETA_ANTI = np.array([
    [27.6239633708124, nan, 0.999999997810276, 16.9774700920608, 13.4821331252169, 17.4467954663834, 0.969802075152177, 37.8284323520572, 28.4713995584883, nan, 1, nan, nan, nan, nan, nan, 1, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [0.999999997810276, nan, 44.6389044846528, 1, 1, 1, 1, 1, 1, nan, 1, nan, nan, nan, nan, nan, 1, nan],
    [16.9774700920608, nan, 1, 1, 1, 1, 1, 1, 2.12224681521582e-10, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [13.4821331252169, nan, 1, 1, 5.65704676687039e-11, 1, 43.8006785504842, 7.58565355874175, 1, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [17.4467954663834, nan, 1, 1, 1, 3.56147513551353, 2.2768635658249e-12, 26.390509335597, 0.121365255672463, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [0.969802075152177, nan, 1, 1, 43.8006785504842, 2.2768635658249e-12, 34.4574083798843, 5.62322055961136e-11, 1, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [37.8284323520572, nan, 1, 1, 7.58565355874175, 26.390509335597, 5.62322055961136e-11, 55.6829793836954, 35.6425218833668, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [28.4713995584883, nan, 1, 2.12224681521582e-10, 1, 0.121365255672463, 1, 35.6425218833668, 5.72558871532628, nan, 1, nan, nan, nan, nan, nan, 1, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [1, nan, 1, nan, nan, nan, nan, nan, 1, nan, 119.699674816712, nan, nan, nan, nan, nan, 1, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [1, nan, 1, nan, nan, nan, nan, nan, 1, nan, 1, nan, nan, nan, nan, nan, 1, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
], dtype=np.float64)

PRE_EXP = np.array([
    [nan, nan, nan, -0.858295878204132, -1.0938094874291e-09, -1.78737309397548e-08, -0.0838261546611818, -2.66886601707483e-09, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, -99.0081347923525, -0.646761012315025, -0.915086549578604, -1.18093112643446, -8.43712628325314, -2.28314625934217, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [-0.858295878204132, nan, -0.646761012315025, -82.6721863355488, -10.5207504472114, -1.35050449002587, -9.39268250563414, -16.8232382033972, -12.6091974932586, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [-1.0938094874291e-09, nan, -0.915086549578604, -10.5207504472114, -0.882584461099775, -99.7156479031364, -0.0130328651215095, -42.5334406719203, -99.0995735110127, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [-1.78737309397548e-08, nan, -1.18093112643446, -1.35050449002587, -99.7156479031364, -44.2237583970833, -12.5202759383878, -5.91399736406338, -12.9831507290928, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [-0.0838261546611818, nan, -8.43712628325314, -9.39268250563414, -0.0130328651215095, -12.5202759383878, -0.288639980040216, -25.870450194609, -18.7104470061527, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [-2.66886601707483e-09, nan, -2.28314625934217, -16.8232382033972, -42.5334406719203, -5.91399736406338, -25.870450194609, -1.58282056298516, -31.6590596973416, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, -12.6091974932586, -99.0995735110127, -12.9831507290928, -18.7104470061527, -31.6590596973416, nan, nan, nan, nan, nan, nan, nan, nan, -76.3033257777515, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, -123.098098160354, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, -76.3033257777515, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
], dtype=np.float64)

ZETA = np.array([
    [nan, nan, nan, 18.1344000135578, 8.56844906043261, 11.1087611106611, 50.2357409779762, 7.5671500952314, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, 5.50152877812295, 3.31389587280406e-12, 2.02755551704376e-14, 9.98468380086182, 2.18354821973587, 1.99132759079248, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [18.1344000135578, nan, 3.31389587280406e-12, 6.47740711268852, 3.66161700080677, 88.297113180888, 8.01270746802158, 4.13965180665037, 5.76541373218234, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [8.56844906043261, nan, 2.02755551704376e-14, 3.66161700080677, 3.48671782959562, 3.21428873712394, 9.98383591800235, 2.74371418048412, 6.10047351801029, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [11.1087611106611, nan, 9.98468380086182, 88.297113180888, 3.21428873712394, 3.33368337046755, 6.94810936582575, 1.31775589055906, 14.5736321802372, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [50.2357409779762, nan, 2.18354821973587, 8.01270746802158, 9.98383591800235, 6.94810936582575, 4.38047773376137, 4.92609196552841, 6.9461838864575, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [7.5671500952314, nan, 1.99132759079248, 4.13965180665037, 2.74371418048412, 1.31775589055906, 4.92609196552841, 2.66993137480615, 19.4566948543194, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, 5.76541373218234, 6.10047351801029, 14.5736321802372, 6.9461838864575, 19.4566948543194, nan, nan, nan, nan, nan, nan, nan, nan, 27.0906233808744, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, 9.99963298939103, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, 27.0906233808744, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
], dtype=np.float64)

R_PARAM = np.array([
    [nan, nan, nan, 0.0015913340501803, 3.77149225649191, 0.0688777920968749, 0.995281956852964, 0.00881287256708145, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, 0.979448597224257, 2.6041321340983, 1.53551435398393, 1.94629973008549, 1.58299419634664, 0.0169341613772597, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [0.0015913340501803, nan, 2.6041321340983, 0.00220129145480174, 1.21000277421539, 1.47289580740841, 1.33523359451779, 1.58590931222507, 1.24712510045945, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [3.77149225649191, nan, 1.53551435398393, 1.21000277421539, 2.40243326681023, 0.763599169596819, 1.82884175715942, 0.881919029239093, 0.608894407221507, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [0.0688777920968749, nan, 1.94629973008549, 1.47289580740841, 0.763599169596819, 0.0196221526216923, 1.07312399726139, 0.400867719435841, 1.09267780582228, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [0.995281956852964, nan, 1.58299419634664, 1.33523359451779, 1.82884175715942, 1.07312399726139, 2.17160799557082, 0.896507908096202, 1.09301691844634, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [0.00881287256708145, nan, 0.0169341613772597, 1.58590931222507, 0.881919029239093, 0.400867719435841, 0.896507908096202, 1.5696675055049, 0.0020386363975639, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, 1.24712510045945, 0.608894407221507, 1.09267780582228, 1.09301691844634, 0.0020386363975639, nan, nan, nan, nan, nan, nan, nan, nan, 1.49729284381904, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, 0.107385590816355, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, 1.49729284381904, nan, nan, nan, nan, nan, nan, nan, nan, nan],
    [nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan],
], dtype=np.float64)

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

# =============================================================================
# Standard Atomic Masses (from NIST/Gaussian)
# =============================================================================

ATOMIC_MASSES: Dict[str, float] = {
    'H': 1.00782503207,
    'He': 4.00260325415,
    'Li': 7.01600455,
    'Be': 9.0121822,
    'B': 11.0093054,
    'C': 12.0000000,
    'N': 14.0030740048,
    'O': 15.99491461956,
    'F': 18.99840322,
    'Ne': 19.9924401754,
    'Na': 22.9897692809,
    'Mg': 23.9850417,
    'Al': 26.98153863,
    'Si': 27.9769265325,
    'P': 30.97376163,
    'S': 31.97207100,
    'Cl': 34.96885268,
    'Ar': 39.9623831225,
}
"""Standard atomic masses in atomic mass units (amu).

These are the most abundant isotope masses used as reference
for isotope effect calculations.
"""

# =============================================================================
# Standard Atomic Masses (from NIST/Gaussian)
# =============================================================================

ATOMIC_MASSES: Dict[str, float] = {
    'H': 1.00782503207,
    'He': 4.00260325415,
    'Li': 7.0160034366,
    'Be': 9.012183065,
    'B': 11.00930536,
    'C': 12.0000000,
    'N': 14.00307400443,
    'O': 15.99491461957,
    'F': 18.99840316273,
    'Ne': 19.9924401762,
    'Na': 22.9897692820,
    'Mg': 23.985041697,
    'Al': 26.98153853,
    'Si': 27.97692653465,
    'P': 30.97376199842,
    'S': 31.9720711744,
    'Cl': 34.968852682,
    'Ar': 39.9623831237,
}
"""Standard atomic masses in atomic mass units (amu)."""

# Common isotope masses for reference
COMMON_ISOTOPES: Dict[str, Dict[str, float]] = {
    'H': {'H': 1.00782503207, 'D': 2.01410177812, 'T': 3.0160492779},
    'C': {'C12': 12.0000000, 'C13': 13.00335483507, 'C14': 14.0032419884},
    'N': {'N14': 14.00307400443, 'N15': 15.00010889888},
    'O': {'O16': 15.99491461957, 'O17': 16.99913175650, 'O18': 17.99915961286},
    'S': {'S32': 31.9720711744, 'S33': 32.9714589098, 'S34': 33.967867004},
    'Cl': {'Cl35': 34.968852682, 'Cl37': 36.965902602},
}
"""Common isotope masses for quick reference."""
