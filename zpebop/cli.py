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
Command-line interface for ZPEBOP.

Supports both ZPEBOP-1 and ZPEBOP-2 models with model selection.
Output format is preserved from the original implementations.
"""

import argparse
import json
import os
from datetime import date, datetime
from pathlib import Path
from typing import Optional
import numpy as np

from .core import ZPECalculator
from .models.base import ZPEResult, BondEnergies

__version__ = '1.0.0'

# Model display names and versions for output
MODEL_DISPLAY_NAMES = {
    'zpebop1': 'ZPE-BOP',
    'zpebop2': 'ZPE-BOP',
}

MODEL_VERSIONS = {
    'zpebop1': '1.0.0',
    'zpebop2': '2.0.0',
}


class NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles NumPy arrays."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class OutputFormatter:
    """Formatter for ZPEBOP output, preserving original format for each model."""
    
    @staticmethod
    def print_header_v1(filepath: str, zpe: float):
        """Print header for ZPEBOP-1 output."""
        today = date.today()
        day_month_year = today.strftime("%d-%B-%Y")
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        
        print(f"\n\n\n{20 * ' '}SUMMARY OF ZPE-BOP CALCULATION\n")
        print(f"{24 * ' '}ZPE-BOP (Version 1.0.0)")
        print(f"{23 * ' '}{day_month_year} {current_time}\n\n")
        print(f'   MINPOP OUTPUT:  {filepath}\n\n')
        print(f'  ZERO-POINT ENERGY (0 K)  =     {zpe:0.3f} KCAL/MOL\n\n')
    
    @staticmethod
    def print_header_v2(filepath: str, zpe: float):
        """Print header for ZPEBOP-2 output."""
        today = date.today()
        day_month_year = today.strftime("%d-%B-%Y")
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        
        print(f"\n\n\n{20 * ' '}SUMMARY OF ZPE-BOP CALCULATION\n")
        print(f"{24 * ' '}ZPE-BOP (Version 2.0.0)")
        print(f"{23 * ' '}{day_month_year} {current_time}\n\n")
        print(f'   MINPOP OUTPUT:  {filepath}\n\n')
        print(f'  ZERO-POINT ENERGY (0 K)  =     {zpe:0.3f} KCAL/MOL\n\n')
    
    @staticmethod
    def print_horizontal_numbers(atoms: np.ndarray, previous: int, 
                                  isotopes: dict = None) -> None:
        """Print column headers for bond energy tables.
        
        Parameters
        ----------
        atoms : np.ndarray
            Array of atom symbols.
        previous : int
            Starting column index.
        isotopes : dict, optional
            Mapping of atom number (1-indexed) to isotope mass.
            Atoms with isotope substitutions will be marked with *.
        """
        print('', end=10 * ' ')
        remaining = len(atoms) - previous
        cols = min(5, remaining)
        
        for i in range(cols):
            col_num = previous + i + 1
            # Add asterisk if this atom has isotope substitution
            marker = '*' if isotopes and col_num in isotopes else ' '
            end = ' ' if i < cols - 1 else '\n'
            print(f'{col_num:>9}{marker}', end=end)
    
    @staticmethod
    def print_lower_diagonal(bond_energies: np.ndarray, atoms: np.ndarray, 
                            previous: int, isotopes: dict = None) -> int:
        """Print lower diagonal elements of bond energy matrix.
        
        Parameters
        ----------
        bond_energies : np.ndarray
            Bond energy matrix.
        atoms : np.ndarray
            Array of atom symbols.
        previous : int
            Starting column index.
        isotopes : dict, optional
            Mapping of atom number (1-indexed) to isotope mass.
            Atoms with isotope substitutions will be marked with *.
        """
        n = len(atoms)
        remaining = n - previous
        cols_this_block = min(5, remaining)
        
        for i in range(previous, n):
            row_num = i + 1
            # Add asterisk if this atom has isotope substitution
            marker = '*' if isotopes and row_num in isotopes else ' '
            atom_label = f"{atoms[i]}{marker}"
            print(f'    {row_num:>2} {atom_label:>5}', end='')
            
            # Number of columns to print for this row
            cols_to_print = min(i - previous + 1, cols_this_block)
            
            for j in range(cols_to_print):
                col_idx = previous + j
                if col_idx <= i:
                    val = bond_energies[i, col_idx]
                    end = '\n' if j == cols_to_print - 1 else '  '
                    print(f'{val:>9.2f}', end=end)
        
        return previous + cols_this_block
    
    @staticmethod
    def print_composite_table(bond_energies: np.ndarray, atoms: np.ndarray,
                             previous: int, isotopes: dict = None) -> int:
        """Print composite table (full matrix view).
        
        Parameters
        ----------
        bond_energies : np.ndarray
            Bond energy matrix.
        atoms : np.ndarray
            Array of atom symbols.
        previous : int
            Starting column index.
        isotopes : dict, optional
            Mapping of atom number (1-indexed) to isotope mass.
        """
        n = len(atoms)
        remaining = n - previous
        cols_this_block = min(5, remaining)
        
        for i in range(n):
            row_num = i + 1
            marker = '*' if isotopes and row_num in isotopes else ' '
            atom_label = f"{atoms[i]}{marker}"
            print(f'    {row_num:>2} {atom_label:>5}', end='')
            
            for j in range(cols_this_block):
                col_idx = previous + j
                val = bond_energies[i, col_idx]
                end = '\n' if j == cols_this_block - 1 else '  '
                print(f'{val:>9.2f}', end=end)
        
        return previous + cols_this_block
    
    @staticmethod
    def print_bond_energies_v1(name: str, bond_energies: np.ndarray, 
                               atoms: np.ndarray, isotopes: dict = None,
                               suffix: str = ''):
        """Print bond energy table for ZPEBOP-1.
        
        Parameters
        ----------
        name : str
            Type of bond energies ('BondEnergies').
        bond_energies : np.ndarray
            Bond energy matrix.
        atoms : np.ndarray
            Array of atom symbols.
        isotopes : dict, optional
            Mapping of atom number (1-indexed) to isotope mass.
        suffix : str, optional
            Suffix to add to title (e.g., '(NORMAL)' or '(ISOTOPE)').
        """
        titles = {
            'BondEnergies': 'Net Vibrational Bond Energies',
        }
        
        title = titles[name].upper()
        if suffix:
            title = f"{title} {suffix}"
        print(f"   {title}\n", end='')
        
        previous = 0
        while previous < len(atoms):
            OutputFormatter.print_horizontal_numbers(atoms, previous, isotopes)
            previous = OutputFormatter.print_lower_diagonal(
                bond_energies, atoms, previous, isotopes
            )
        print('\n')
        total = np.sum(bond_energies)
        print(f'   TOTAL NET VIB. BOND ENERGY =     {total:0.2f} KCAL/MOL', end='')
        print('\n\n')
    
    @staticmethod
    def print_bond_energies_v2(name: str, bond_energies: np.ndarray, 
                               atoms: np.ndarray, isotopes: dict = None,
                               suffix: str = ''):
        """Print bond energy table for ZPEBOP-2.
        
        Parameters
        ----------
        name : str
            Type of bond energies ('GrossBond', 'NetBond', 'CompositeTable').
        bond_energies : np.ndarray
            Bond energy matrix.
        atoms : np.ndarray
            Array of atom symbols.
        isotopes : dict, optional
            Mapping of atom number (1-indexed) to isotope mass.
        suffix : str, optional
            Suffix to add to title (e.g., '(NORMAL)' or '(ISOTOPE)').
        """
        titles = {
            'GrossBond': 'Gross Total Bond Energies',
            'NetBond': 'Net Total Energies',
            'CompositeTable': ('Composite Table:    0            Eij(gross)\n' + 
                              23 * ' ' + 'Eji(net)     0')
        }
        
        energy_labels = {
            'GrossBond': 'Total Gross Energy',
            'NetBond': 'Total Net Energy'
        }
        
        title = titles[name].upper() if name != 'CompositeTable' else titles[name]
        if suffix and name != 'CompositeTable':
            title = f"{title} {suffix}"
        print(f"   {title}\n", end='')
        
        if name in ['GrossBond', 'NetBond']:
            previous = 0
            while previous < len(atoms):
                OutputFormatter.print_horizontal_numbers(atoms, previous, isotopes)
                previous = OutputFormatter.print_lower_diagonal(
                    bond_energies, atoms, previous, isotopes
                )
            print('\n')
            total = np.sum(bond_energies)
            print(f'   {energy_labels[name].upper()}     =     {total:0.2f} KCAL/MOL', 
                  end='')
        
        elif name == 'CompositeTable':
            previous = 0
            while previous < len(atoms):
                OutputFormatter.print_horizontal_numbers(atoms, previous, isotopes)
                previous = OutputFormatter.print_composite_table(
                    bond_energies, atoms, previous, isotopes
                )
            print('\n')
        
        print('\n\n')
    
    @staticmethod
    def print_sorted_bonds_v1(labels: np.ndarray, energies: np.ndarray):
        """Print sorted bond energies table for ZPEBOP-1."""
        print(f'  SORTED NET VIB. BONDING ENERGIES (LOWEST TO HIGHEST)\n')
        print(f'  ----------------------------')
        print(f'     BOND       BOND ENERGIES ')
        print(f'   IDENTITY       (KCAL/MOL)  ')
        print(f'  ----------------------------')
        
        for label, energy in zip(labels, energies):
            print(f'  {label:>9}         {energy:<5.2f}')
        
        print(f'  ----------------------------')
        print('')
    
    @staticmethod
    def print_sorted_bonds_v2(labels: np.ndarray, energies: np.ndarray):
        """Print sorted bond energies table for ZPEBOP-2."""
        print(f'  SORTED NET VIB. BONDING ENERGIES (LOWEST TO HIGHEST)\n')
        print(f'  ----------------------------')
        print(f'     BOND       BOND ENERGIES ')
        print(f'   IDENTITY       (KCAL/MOL)  ')
        print(f'  ----------------------------')
        
        for label, energy in zip(labels, energies):
            print(f'  {label:>9}         {energy:<5.2f}')
        
        print(f'  ----------------------------')
        print('')
    
    @staticmethod
    def print_isotope_header(isotopes: dict, atoms: np.ndarray):
        """Print isotope substitution information."""
        print(f'   ISOTOPE SUBSTITUTIONS:')
        for atom_num, mass in sorted(isotopes.items()):
            symbol = atoms[atom_num - 1]
            # Check for common isotope names
            label = f"{symbol}{atom_num}"
            if symbol == 'H':
                if abs(mass - 2.014) < 0.01:
                    label = f"D{atom_num}"
                elif abs(mass - 3.016) < 0.01:
                    label = f"T{atom_num}"
            print(f'     {label}: {mass:.6f} amu')
        print('')
    
    @staticmethod
    def print_isotope_zpe_comparison(zpe_normal: float, zpe_isotope: float):
        """Print comparison of normal and isotope ZPE."""
        delta = zpe_normal - zpe_isotope
        ratio = zpe_isotope / zpe_normal if zpe_normal > 0 else 1.0
        
        print(f'  ZERO-POINT ENERGY COMPARISON:')
        print(f'    Normal ZPE (0 K)   =     {zpe_normal:0.3f} KCAL/MOL')
        print(f'    Isotope ZPE (0 K)  =     {zpe_isotope:0.3f} KCAL/MOL')
        print(f'    ΔZPE (Normal - Iso)=     {delta:0.3f} KCAL/MOL')
        print(f'    Ratio (Iso/Normal) =     {ratio:0.6f}')
        print('\n')
    
    @staticmethod
    def print_isotope_comparison_table(name: str, energies_normal: np.ndarray,
                                        energies_isotope: np.ndarray, 
                                        atoms: np.ndarray, isotopes: dict):
        """Print bond energy comparison table with normal vs isotope values.
        
        Only shows bonds that involve isotope-substituted atoms (bonds that change).
        Includes all bonds (bonding and anti-bonding) with no threshold.
        
        Parameters
        ----------
        name : str
            Type of energies ('NetBond' or 'GrossBond').
        energies_normal : np.ndarray
            Normal (non-isotope) bond energy matrix.
        energies_isotope : np.ndarray
            Isotope-corrected bond energy matrix.
        atoms : np.ndarray
            Array of atom symbols.
        isotopes : dict
            Mapping of atom number (1-indexed) to isotope mass.
        """
        titles = {
            'NetBond': 'NET VIBRATIONAL BOND ENERGIES',
            'GrossBond': 'GROSS VIBRATIONAL BOND ENERGIES',
        }
        
        print(f"   {titles[name]} (ISOTOPE COMPARISON)\n")
        print(f"   {'Bond':<12} {'Normal':<12} {'Isotope':<12} {'Ratio':<10}")
        print(f"   {'-'*46}")
        
        n = len(atoms)
        for i in range(1, n):
            for j in range(i):
                # Only show bonds involving isotope-substituted atoms
                if (j + 1) not in isotopes and (i + 1) not in isotopes:
                    continue
                
                e_normal = energies_normal[i, j]
                e_isotope = energies_isotope[i, j]
                ratio = e_isotope / e_normal if abs(e_normal) > 1e-10 else 1.0
                
                # Add asterisk markers for isotope atoms
                marker_j = '*' if (j + 1) in isotopes else ''
                marker_i = '*' if (i + 1) in isotopes else ''
                bond_label = f"{atoms[j]}{j+1}{marker_j}-{atoms[i]}{i+1}{marker_i}"
                
                print(f"   {bond_label:<12} {e_normal:<12.3f} {e_isotope:<12.3f} {ratio:<10.4f}")
        
        print(f"   {'-'*46}")
        print('\n')


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for ZPEBOP."""
    parser = argparse.ArgumentParser(
        description='Compute ZPE and bond energies using ZPEBOP models'
    )
    parser.add_argument(
        '-f', 
        required=True, 
        type=str,
        help='Name of the Gaussian Hartree-Fock output file'
    )
    parser.add_argument(
        '--model',
        type=str,
        choices=['zpebop1', 'zpebop2'],
        default='zpebop2',
        help='ZPEBOP model to use (default: zpebop2)'
    )
    parser.add_argument(
        '--be',
        action='store_true',
        help='Compute ZPEBOP vibrational bond energies'
    )
    parser.add_argument(
        '--sort',
        action='store_true',
        help='Sort the ZPEBOP bond energies (lowest to highest)'
    )
    parser.add_argument(
        '--isotope',
        action='append',
        type=str,
        metavar='ATOM:MASS',
        help='Isotope substitution as atom_number:mass (e.g., 1:2.014102 for D). '
             'Can be specified multiple times for multiple substitutions.'
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Print isotope comparison tables showing normal vs isotope bond energies. '
             'Only used with --isotope and --be flags.'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Save the output to JSON format'
    )
    parser.add_argument(
        '-o',
        type=str,
        default='bopout.json',
        help='Output JSON filename (default: bopout.json)'
    )
    
    return parser


def parse_isotopes(isotope_args: list) -> dict:
    """
    Parse isotope arguments into a dictionary.
    
    Parameters
    ----------
    isotope_args : list
        List of strings in format "atom_number:mass"
    
    Returns
    -------
    dict
        Mapping of atom number (int) to mass (float)
    """
    if not isotope_args:
        return {}
    
    isotopes = {}
    for arg in isotope_args:
        try:
            atom_str, mass_str = arg.split(':')
            atom_num = int(atom_str)
            mass = float(mass_str)
            isotopes[atom_num] = mass
        except ValueError:
            raise ValueError(
                f"Invalid isotope format '{arg}'. "
                f"Expected format: atom_number:mass (e.g., 1:2.014102)"
            )
    
    return isotopes


def generate_comparison_json(energies_normal: np.ndarray, energies_isotope: np.ndarray,
                              atoms: np.ndarray, isotopes: dict) -> dict:
    """
    Generate isotope comparison data for JSON output.
    
    Only includes bonds that involve isotope-substituted atoms (bonds that change).
    Includes all bonds (bonding and anti-bonding) with no threshold.
    
    Parameters
    ----------
    energies_normal : np.ndarray
        Normal bond energy matrix.
    energies_isotope : np.ndarray
        Isotope-corrected bond energy matrix.
    atoms : np.ndarray
        Array of atom symbols.
    isotopes : dict
        Mapping of atom number (1-indexed) to isotope mass.
    
    Returns
    -------
    dict
        Comparison data with bonds, normal values, isotope values, and ratios.
    """
    bonds = []
    normal_values = []
    isotope_values = []
    ratios = []
    
    n = len(atoms)
    for i in range(1, n):
        for j in range(i):
            # Only include bonds involving isotope-substituted atoms
            if (j + 1) not in isotopes and (i + 1) not in isotopes:
                continue
            
            e_normal = energies_normal[i, j]
            e_isotope = energies_isotope[i, j]
            
            # Add asterisk markers for isotope atoms
            marker_j = '*' if (j + 1) in isotopes else ''
            marker_i = '*' if (i + 1) in isotopes else ''
            bond_label = f"{atoms[j]}{j+1}{marker_j}-{atoms[i]}{i+1}{marker_i}"
            
            ratio = e_isotope / e_normal if abs(e_normal) > 1e-10 else 1.0
            
            bonds.append(bond_label)
            normal_values.append(float(e_normal))
            isotope_values.append(float(e_isotope))
            ratios.append(float(ratio))
    
    return {
        'bonds': bonds,
        'normal': normal_values,
        'isotope': isotope_values,
        'ratio': ratios
    }


def run_calculation(args):
    """Run ZPEBOP calculation and print results."""
    # Initialize calculator with selected model
    calc = ZPECalculator(args.f, model=args.model)
    
    # Parse isotopes if provided
    isotopes = parse_isotopes(args.isotope) if args.isotope else {}
    
    # Print header based on model
    filepath = os.path.abspath(args.f)
    
    if isotopes:
        # Isotope calculation
        result = calc.compute_zpe_isotope(isotopes)
        
        if args.model == 'zpebop1':
            OutputFormatter.print_header_v1(filepath, result.total_zpe)
        else:
            OutputFormatter.print_header_v2(filepath, result.total_zpe)
        
        # Print isotope information
        OutputFormatter.print_isotope_header(isotopes, calc.atoms)
        OutputFormatter.print_isotope_zpe_comparison(result.total_zpe_normal, 
                                                      result.total_zpe)
        
        # Initialize JSON dictionary if needed
        if args.json:
            json_dict = {
                'method': MODEL_DISPLAY_NAMES[args.model],
                'version': MODEL_VERSIONS[args.model],
                'MINPOP output file': filepath,
                'date': date.today().strftime("%d-%B-%Y"),
                'time': datetime.now().strftime("%H:%M:%S"),
                'isotope calculation': True,
                'isotope substitutions': {str(k): v for k, v in isotopes.items()},
                'zero point energy (normal)': result.total_zpe_normal,
                'zero point energy (isotope)': result.total_zpe,
                'delta ZPE': result.zpe_difference,
                'ZPE ratio': result.zpe_ratio
            }
        
        # Print bond energies if requested
        if args.be:
            normal_be, isotope_be = calc.compute_bond_energies_isotope(isotopes)
            
            if args.model == 'zpebop1':
                # Print isotope-corrected matrix with asterisks on substituted atoms
                OutputFormatter.print_bond_energies_v1(
                    'BondEnergies', isotope_be.net, calc.atoms,
                    isotopes=isotopes
                )
                # Print comparison table if requested
                if args.compare:
                    OutputFormatter.print_isotope_comparison_table(
                        'NetBond', normal_be.net, isotope_be.net,
                        calc.atoms, isotopes
                    )
                if args.json:
                    json_dict['bond energies'] = {
                        'net': {'normal': normal_be.net, 'isotope': isotope_be.net}
                    }
                    if args.compare:
                        json_dict['net bond energies (isotope comparison)'] = \
                            generate_comparison_json(normal_be.net, isotope_be.net, 
                                                     calc.atoms, isotopes)
            else:
                # Print isotope-corrected gross and net matrices with asterisks
                OutputFormatter.print_bond_energies_v2(
                    'GrossBond', isotope_be.gross, calc.atoms,
                    isotopes=isotopes
                )
                OutputFormatter.print_bond_energies_v2(
                    'NetBond', isotope_be.net, calc.atoms,
                    isotopes=isotopes
                )
                # Print comparison tables if requested
                if args.compare:
                    OutputFormatter.print_isotope_comparison_table(
                        'GrossBond', normal_be.gross, isotope_be.gross,
                        calc.atoms, isotopes
                    )
                    OutputFormatter.print_isotope_comparison_table(
                        'NetBond', normal_be.net, isotope_be.net,
                        calc.atoms, isotopes
                    )
                if args.json:
                    json_dict['bond energies'] = {
                        'gross': {'normal': normal_be.gross, 'isotope': isotope_be.gross},
                        'net': {'normal': normal_be.net, 'isotope': isotope_be.net}
                    }
                    if args.compare:
                        json_dict['gross bond energies (isotope comparison)'] = \
                            generate_comparison_json(normal_be.gross, isotope_be.gross,
                                                     calc.atoms, isotopes)
                        json_dict['net bond energies (isotope comparison)'] = \
                            generate_comparison_json(normal_be.net, isotope_be.net,
                                                     calc.atoms, isotopes)
        
        # Print sorted bonds if requested
        if args.sort:
            if not args.be:
                normal_be, isotope_be = calc.compute_bond_energies_isotope(isotopes)
            
            if args.model == 'zpebop1':
                labels, energies = calc.sort_bond_energies(isotope_be.net, isotopes=isotopes)
                OutputFormatter.print_sorted_bonds_v1(labels, energies)
            else:
                labels, energies = calc.sort_bond_energies(isotope_be.net, isotopes=isotopes)
                OutputFormatter.print_sorted_bonds_v2(labels, energies)
            
            if args.json:
                json_dict['sorted net bond energies'] = {
                    'sorted bonds': labels.tolist(),
                    'bond energies': energies.tolist()
                }
    
    else:
        # Normal calculation (no isotopes)
        result = calc.compute_zpe()
        
        if args.model == 'zpebop1':
            OutputFormatter.print_header_v1(filepath, result.total_zpe)
        else:
            OutputFormatter.print_header_v2(filepath, result.total_zpe)
        
        # Initialize JSON dictionary if needed
        if args.json:
            json_dict = {
                'method': MODEL_DISPLAY_NAMES[args.model],
                'version': MODEL_VERSIONS[args.model],
                'MINPOP output file': filepath,
                'date': date.today().strftime("%d-%B-%Y"),
                'time': datetime.now().strftime("%H:%M:%S"),
                'zero point energy': result.total_zpe
            }
        
        # Print bond energies if requested
        if args.be:
            bond_energies = calc.compute_bond_energies()
            
            if args.model == 'zpebop1':
                OutputFormatter.print_bond_energies_v1('BondEnergies', 
                                                       bond_energies.net, 
                                                       calc.atoms)
                if args.json:
                    json_dict['bond energies'] = {
                        'net': bond_energies.net
                    }
            else:
                OutputFormatter.print_bond_energies_v2('GrossBond', bond_energies.gross, 
                                                       calc.atoms)
                OutputFormatter.print_bond_energies_v2('NetBond', bond_energies.net,
                                                       calc.atoms)
                OutputFormatter.print_bond_energies_v2('CompositeTable', bond_energies.composite,
                                                       calc.atoms)
                
                if args.json:
                    json_dict['bond energies'] = {
                        'gross': bond_energies.gross,
                        'net': bond_energies.net,
                        'composite table': bond_energies.composite
                    }
        
        # Print sorted bonds if requested
        if args.sort:
            if not args.be:
                bond_energies = calc.compute_bond_energies()
            
            if args.model == 'zpebop1':
                labels, energies = calc.sort_bond_energies(bond_energies.net)
                OutputFormatter.print_sorted_bonds_v1(labels, energies)
            else:
                labels, energies = calc.sort_bond_energies(bond_energies.net)
                OutputFormatter.print_sorted_bonds_v2(labels, energies)
            
            if args.json:
                json_dict['sorted net bond energies'] = {
                    'sorted bonds': labels.tolist(),
                    'bond energies': energies.tolist()
                }
    
    # Write JSON if requested
    if args.json:
        output_file = args.o
        with open(output_file, 'w') as f:
            json.dump(json_dict, f, indent=4, cls=NumpyJSONEncoder)


def main():
    """Main entry point for ZPEBOP CLI."""
    parser = create_parser()
    args = parser.parse_args()
    run_calculation(args)


if __name__ == '__main__':
    main()
