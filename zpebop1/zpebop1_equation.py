# MIT License
# 
# Copyright (c) 2024, Barbaro Zulueta
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

"""Computes the zpve energy contributions
   E_zpve :: covalent energy (for sigma and pi)
   conditional_statements :: conditional protocols
   zpve :: total zpve energy in Hartrees
   """

import numpy as np
from . import zpebop1_params as param

# BEBOP sub-equations
def E_zpve(bo, beta):
    """Calculate the zpve energy from extended-Hückel 
       theory
    
       Parameters
       ----------
       bo: :obj:'numpy.ndarray'
           Mulliken bond order
       beta: :obj:'numpy.float64'
           Extend-Hückel theory parameter
           for Mulliken bond order
       
       Returns
       -------
       zpve: :obj:'numpy.ndarray'     
           total zpve bond energy
       """
    
    zpve = 2 * beta * bo
    return zpve

def conditional_statements(pair_atoms, bo):
    """Condition protocols for zpve
    
       Parameters
       ----------
       pair_atoms: :obj:'numpy.ndarray'
           array showing the two pair of atoms
       bo: :obj:'numpy.float64'
           Mulliken bond order
           
       Returns
       -------
       corrections: :obj:'int'     
           the correction value: 
               True -> 0 > bo
               False -> bo > 0
       """
    # Conditional protocols for computing zpe bond energies
    if 0 > bo:
        corrections = True # anti-bonding 
    elif bo >= 0:
        corrections = False # bonding
    return corrections

# MAIN ZPVE EQUATION
def zpve(atoms, bo):
    """Calculate zpe energy and zpe bond 
    
       Parameters
       ----------
       atoms: :obj:'numpy.ndarray'
           array showing the atoms
       bo: :obj:'numpy.ndarray'
           Mulliken bond order
           
       Returns
       -------
       zpve: :obj:'numpy.float64'
           the total zpve in Hartrees 
       bond_zpve: :obj:'numpy.ndarray'
           the vibrational bond energies at 0 K in Hartrees
       """
    size = atoms.shape[0]
    bond_zpve = np.zeros((size,size), dtype = np.float64) # coulombs law for ionic interactions
    for l in range(1,size):
        for n in range(l):
            # important properties
            index = (l,n)
            pair_atoms = np.array([atoms[l], atoms[n]])
            bo_pair = bo[l][n]
            anti_corr = conditional_statements(pair_atoms, bo_pair)
            
            # fitting parameters
            beta = param.zpve_pair(pair_atoms, anti_corr)

            # extended Hückel covalent bonding for sigma and pi bonding 
            bond_zpve[l][n] = E_zpve(bo_pair, beta)
                       
    # Calculate total zero-point energy
    zpve = np.sum(bond_zpve)
    return (zpve, bond_zpve) 

def zpebop1_bond_energy(zpve, bond_zpve):
    """Calculate the individual vibrational bond energy contributions
    
    Parameters
    ----------
    zpve: :obj:'numpy.float64'
        zero-point vibrational energy at 0K in a molecule
    bond_zpve: :obj:'numpy.ndarray'
        the vibrational bond energies at 0 K
        
    Returns
    -------
    E_gross: :obj:'numpy.ndarray' 
        gross vibrational bond energies (i.e., E_gross = E_Huckel,vib)
    E_net: :obj:'numpy.ndarray'
        net vibrational bond energies (i.e., E_net = E_gross)
    CompositeTable: :obj:'numpy.ndarray'
        matrix showing gross vibrational bond energies(upper diagonal elements) 
        and net vibrational bond energies (lower diagonal elements)
    """
    
    size = bond_zpve.shape[0]
    E_gross = bond_zpve
    E_net = bond_zpve
    CompositeTable = np.zeros((size,size), dtype = np.float64)
    
    for i in range(1,size):
        for j in range(i):
            CompositeTable[j][i] = E_gross[i][j] # gross energies for 
                                                 # the upper diagonal matrices
            CompositeTable[i][j] = E_net[i][j] # net bond energies for
                                               # the lower diagonal matrices
    
    return (E_gross, E_net, CompositeTable)
