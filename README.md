# ZPEBOP

**Zero-Point Energies from Bond Orders and Populations**

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

ZPEBOP is a high-performance Python package for computing molecular zero-point vibrational energies (ZPE) from Mulliken bond orders obtained from ROHF calculations.

## Available Models

| Model | Description | Terms |
|-------|-------------|-------|
| **ZPEBOP-1** | Harmonic approximation | $E = 2\beta\|P\|$ |
| **ZPEBOP-2** | Full model (default) | Harmonic + Anharmonic + Three-body |

## Features

- **Multiple models**: Choose between ZPEBOP-1 (fast) and ZPEBOP-2 (accurate)
- **Unified interface**: Same API for all models
- **Isotope corrections**: Compute kinetic isotope effects on ZPE
- **Bond-resolved analysis**: Decompose ZPE into individual bond contributions
- **High performance**: Vectorized NumPy operations with pre-computed parameters
- **Self-contained**: All parameters bundled - no external files needed
- **Command-line interface**: Quick calculations from the terminal

## Installation

```bash
git clone https://github.com/keithgroup/zpebop-qc.git
cd zpebop-qc/zpebop
pip install -e .
```

### Dependencies

- Python >= 3.7
- NumPy >= 1.20.0

## Quick Start

### Python API

```python
from zpebop import ZPECalculator

# Use ZPEBOP-2 (default, most accurate)
calc = ZPECalculator("molecule.out")
result = calc.compute_zpe()
print(f"ZPE = {result.total_zpe:.3f} kcal/mol")

# Use ZPEBOP-1 (harmonic only)
calc = ZPECalculator("molecule.out", model="zpebop1")
result = calc.compute_zpe()
print(f"ZPE = {result.total_zpe:.3f} kcal/mol")

# Get bond energies
bond_energies = calc.compute_bond_energies()
print(f"Total: {bond_energies.gross.sum():.3f} kcal/mol")

# Sort bonds by energy
labels, energies = calc.sort_bond_energies(bond_energies.gross)
for label, e in zip(labels[-5:], energies[-5:]):
    print(f"{label}: {e:.2f} kcal/mol")
```

### Command Line

```bash
# Use ZPEBOP-2 (default)
zpebop -f molecule.out

# Use ZPEBOP-1
zpebop -f molecule.out --model zpebop1

# With bond energy tables
zpebop -f molecule.out --be

# With sorted bonds and JSON output
zpebop -f molecule.out --be --sort --json
```

## Isotope Corrections

ZPEBOP supports isotope effect calculations for studying kinetic isotope effects (KIE).
The correction uses the harmonic oscillator approximation:

$$BE_{\text{isotope}} = BE_{\text{normal}} \times \sqrt{\frac{\mu_{\text{normal}}}{\mu_{\text{isotope}}}}$$

where $\mu = \frac{m_1 \times m_2}{m_1 + m_2}$ is the reduced mass.

### Python API

```python
from zpebop import ZPECalculator

calc = ZPECalculator("molecule.out")

# Define isotopes: atom_number (1-indexed) -> mass (amu)
isotopes = {
    1: 2.014102,   # Atom 1 is deuterium (D)
    7: 13.00335,   # Atom 7 is carbon-13
}

# Compute with isotope corrections
result = calc.compute_zpe_isotope(isotopes)

# Access results
print(f"Normal ZPE:  {result.total_zpe_normal:.3f} kcal/mol")
print(f"Isotope ZPE: {result.total_zpe:.3f} kcal/mol")
print(f"Delta ZPE:   {result.zpe_difference:.3f} kcal/mol")
print(f"Ratio:       {result.zpe_ratio:.6f}")

# Get isotope-corrected bond energies
normal_be, isotope_be = calc.compute_bond_energies_isotope(isotopes)
```

### Command Line

```bash
# Single isotope substitution (deuterium at atom 1)
zpebop -f molecule.out --isotope 1:2.014102

# Multiple isotope substitutions
zpebop -f molecule.out --isotope 1:2.014102 --isotope 7:13.00335

# With bond energy tables (asterisks mark substituted atoms)
zpebop -f molecule.out --isotope 1:2.014102 --be

# With isotope comparison tables (normal vs isotope)
zpebop -f molecule.out --isotope 1:2.014102 --be --compare

# With JSON output
zpebop -f molecule.out --isotope 1:2.014102 --json
```

### Common Isotope Masses

| Isotope | Mass (amu) | Usage |
|---------|------------|-------|
| Deuterium (D) | 2.014102 | `--isotope N:2.014102` |
| Tritium (T) | 3.016049 | `--isotope N:3.016049` |
| Carbon-13 | 13.00335 | `--isotope N:13.00335` |
| Carbon-14 | 14.00324 | `--isotope N:14.00324` |
| Nitrogen-15 | 15.00011 | `--isotope N:15.00011` |
| Oxygen-18 | 17.99916 | `--isotope N:17.99916` |

### Isotope Output Format

```
   ISOTOPE SUBSTITUTIONS:
     D1: 2.014102 amu

  ZERO-POINT ENERGY COMPARISON:
    Normal ZPE (0 K)   =     64.044 KCAL/MOL
    Isotope ZPE (0 K)  =     62.293 KCAL/MOL
    Delta ZPE          =      1.752 KCAL/MOL
    Ratio (Iso/Normal) =     0.972651
```

### Isotope Comparison Tables (--compare)

With the `--compare` flag, comparison tables show all bonds involving the isotope-substituted atom:

```
   NET VIBRATIONAL BOND ENERGIES (ISOTOPE COMPARISON)

   Bond         Normal       Isotope      Ratio     
   ----------------------------------------------
   C1-H7*       5.067        3.720        0.7342    
   C2-H7*       0.285        0.209        0.7342    
   O3-H7*       0.002        0.001        0.7280    
   C4-H7*       0.010        0.007        0.7342    
   C5-H7*       0.000        0.000        0.7342    
   H6-H7*       0.003        0.003        0.8661    
   H7*-H8       0.368        0.319        0.8661    
   H7*-H9       0.059        0.051        0.8661    
   H7*-H10      0.001        0.000        0.8661    
   H7*-H11      0.000        0.000        0.8661    
   H7*-H12      0.000        0.000        0.8661    
   H7*-H13      0.000        0.000        1.0000    
   ----------------------------------------------
```

## Gaussian Input Requirements

ZPEBOP requires Gaussian output from ROHF/CBSB3 calculations with population analysis:

```
# ROHF/CBSB3 Pop=(Full) IOp(6/27=122)
```

## Output Format

### ZPEBOP-1 Output

```
                    SUMMARY OF ZPE-BOP CALCULATION

                        ZPE-BOP (Version 1.0.0)
                       17-January-2026 12:00:00


   MINPOP OUTPUT:  /path/to/molecule.out

  ZERO-POINT ENERGY (0 K)  =     63.801 KCAL/MOL
```

### ZPEBOP-2 Output

```
                    SUMMARY OF ZPE-BOP CALCULATION

                        ZPE-BOP (Version 2.0.0)
                       17-January-2026 12:00:00


   MINPOP OUTPUT:  /path/to/molecule.out

  ZERO-POINT ENERGY (0 K)  =     64.044 KCAL/MOL
```

## API Reference

### ZPECalculator

```python
from zpebop import ZPECalculator

# Initialize with model selection
calc = ZPECalculator(source, model='zpebop2')

# Available methods
result = calc.compute_zpe()           # Returns ZPEResult
energies = calc.compute_bond_energies()  # Returns BondEnergies
labels, values = calc.sort_bond_energies(matrix)

# Isotope methods
iso_result = calc.compute_zpe_isotope(isotopes)  # Returns IsotopeZPEResult
normal_be, iso_be = calc.compute_bond_energies_isotope(isotopes)
```

### ZPEResult

```python
result.total_zpe      # Total ZPE in kcal/mol
result.two_body       # Two-body contributions matrix
result.three_body_decomp  # Three-body decomposed to pairs
result.gross          # Alias for two_body
result.net            # two_body + three_body_decomp
result.model          # 'zpebop1' or 'zpebop2'
```

### IsotopeZPEResult

```python
result.total_zpe          # Isotope-corrected total ZPE
result.total_zpe_normal   # Normal (uncorrected) total ZPE
result.zpe_difference     # Normal - Isotope (Delta ZPE)
result.zpe_ratio          # Isotope/Normal ratio
result.two_body           # Isotope-corrected two-body matrix
result.two_body_normal    # Normal two-body matrix
result.correction_factors # Matrix of correction factors
result.isotopes           # Dict of {atom_num: mass}
```

### BondEnergies

```python
energies.gross        # Gross (two-body) matrix
energies.net          # Net (with three-body) matrix
energies.composite    # Combined table
```

## Supported Elements

| Period | Elements |
|--------|----------|
| 1 | H |
| 2 | Li, Be, B, C, N, O, F |
| 3 | Na, Cl |

Three-body terms (ZPEBOP-2) are available for B, C, N, O pairs.

## Citation

If you use ZPEBOP in your research, please cite:

```bibtex
@article{zulueta2025zpebop,
  title={Zero-point energies from bond orders and populations relationships},
  author={Zulueta, Barbaro and Rude, Colin D. and Mangiardi, Jesse A. and 
          Petersson, George A. and Keith, John A.},
  journal={The Journal of Chemical Physics},
  volume={162},
  number={8},
  pages={084102},
  year={2025},
  doi={10.1063/5.0238831}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.
