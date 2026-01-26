# ZPEBOP

**Zero-Point Energies from Bond Orders and Populations**

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

ZPEBOP is a high-performance Python package for computing molecular zero-point vibrational energies (ZPE) from Mulliken bond orders obtained from ROHF calculations.

## Available Models

| Model | Description | Description |
|-------|-------------|-------|
| **ZPEBOP-1** | Harmonic approximation | Harmonic |
| **ZPEBOP-2** | Full model (default) | Harmonic + Anharmonic + Three-body |

## Features

- **Multiple models**: Choose between ZPEBOP-1 (fast) and ZPEBOP-2 (accurate)
- **Unified interface**: Same API for all models
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
