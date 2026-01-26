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
    def print_horizontal_numbers(atoms: np.ndarray, previous: int) -> None:
        """Print column headers for bond energy tables.
        
        Parameters
        ----------
        atoms : np.ndarray
            Array of atom symbols.
        previous : int
            Starting column index.
        """
        print('', end=10 * ' ')
        remaining = len(atoms) - previous
        cols = min(5, remaining)
        
        for i in range(cols):
            col_num = previous + i + 1
            end = ' ' if i < cols - 1 else '\n'
            print(f'{col_num:>9} ', end=end)
    
    @staticmethod
    def print_lower_diagonal(bond_energies: np.ndarray, atoms: np.ndarray, 
                            previous: int) -> int:
        """Print lower diagonal elements of bond energy matrix.
        
        Parameters
        ----------
        bond_energies : np.ndarray
            Bond energy matrix.
        atoms : np.ndarray
            Array of atom symbols.
        previous : int
            Starting column index.
        """
        n = len(atoms)
        remaining = n - previous
        cols_this_block = min(5, remaining)
        
        for i in range(previous, n):
            row_num = i + 1
            atom_label = f"{atoms[i]} "
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
                             previous: int) -> int:
        """Print composite table (full matrix view).
        
        Parameters
        ----------
        bond_energies : np.ndarray
            Bond energy matrix.
        atoms : np.ndarray
            Array of atom symbols.
        previous : int
            Starting column index.
        """
        n = len(atoms)
        remaining = n - previous
        cols_this_block = min(5, remaining)
        
        for i in range(n):
            row_num = i + 1
            atom_label = f"{atoms[i]} "
            print(f'    {row_num:>2} {atom_label:>5}', end='')
            
            for j in range(cols_this_block):
                col_idx = previous + j
                val = bond_energies[i, col_idx]
                end = '\n' if j == cols_this_block - 1 else '  '
                print(f'{val:>9.2f}', end=end)
        
        return previous + cols_this_block
    
    @staticmethod
    def print_bond_energies_v1(name: str, bond_energies: np.ndarray, 
                               atoms: np.ndarray, suffix: str = ''):
        """Print bond energy table for ZPEBOP-1.
        
        Parameters
        ----------
        name : str
            Type of bond energies ('BondEnergies').
        bond_energies : np.ndarray
            Bond energy matrix.
        atoms : np.ndarray
            Array of atom symbols.
        suffix : str, optional
            Suffix to add to title.
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
            OutputFormatter.print_horizontal_numbers(atoms, previous)
            previous = OutputFormatter.print_lower_diagonal(
                bond_energies, atoms, previous
            )
        print('\n')
        total = np.sum(bond_energies)
        print(f'   TOTAL NET VIB. BOND ENERGY =     {total:0.2f} KCAL/MOL', end='')
        print('\n\n')
    
    @staticmethod
    def print_bond_energies_v2(name: str, bond_energies: np.ndarray, 
                               atoms: np.ndarray, suffix: str = ''):
        """Print bond energy table for ZPEBOP-2.
        
        Parameters
        ----------
        name : str
            Type of bond energies ('GrossBond', 'NetBond', 'CompositeTable').
        bond_energies : np.ndarray
            Bond energy matrix.
        atoms : np.ndarray
            Array of atom symbols.
        suffix : str, optional
            Suffix to add to title.
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
                OutputFormatter.print_horizontal_numbers(atoms, previous)
                previous = OutputFormatter.print_lower_diagonal(
                    bond_energies, atoms, previous
                )
            print('\n')
            total = np.sum(bond_energies)
            print(f'   {energy_labels[name].upper()}     =     {total:0.2f} KCAL/MOL', 
                  end='')
        
        elif name == 'CompositeTable':
            previous = 0
            while previous < len(atoms):
                OutputFormatter.print_horizontal_numbers(atoms, previous)
                previous = OutputFormatter.print_composite_table(
                    bond_energies, atoms, previous
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


def run_calculation(args):
    """Run ZPEBOP calculation and print results."""
    # Initialize calculator with selected model
    calc = ZPECalculator(args.f, model=args.model)
    
    # Print header based on model
    filepath = os.path.abspath(args.f)
    
    # Normal calculation
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
