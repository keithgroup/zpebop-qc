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
   zpve :: total zpve energy in kcal/mol
   """

import numpy as np
from . import zpebop2_params as param

# BEBOP sub-equations
def E_harm(bo, beta):
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
    
    zpve = 2 * beta * np.abs(bo)
    return zpve

def conditional_statements(pair_atoms, bo, size):
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
    # Conditional protocols for computing zpve bond energies
    if 0 > bo and size > 2:
        corrections = True # anti-bonding 
    else:
        corrections = False # bonding
    return corrections

def E_anharm(distance, pre_exp, zeta, R_param):
    if pre_exp == 'None':
        value = 0
    else:
        value = pre_exp * np.exp(-zeta * (distance - R_param))
    return value

def calc_cosines(R_ij, R_ik, R_jk):
    cos_ij = ((R_ik**2 + R_jk**2) - R_ij**2)/(2 * R_ik * R_jk)
    if cos_ij > 1:
        cos_ij = 1
    if cos_ij < -1:
        cos_ij = -1

    # Cosine between atoms i and k
    cos_ik = ((R_ij**2 + R_jk**2) - R_ik**2)/(2 * R_ij * R_jk)
    if cos_ik > 1:
        cos_ik = 1
    if cos_ik < -1:
        cos_ik = -1
        
    # Cosine between atoms j and k
    cos_jk = ((R_ij**2 + R_ik**2) - R_jk**2)/(2 * R_ij * R_ik)
    if cos_jk > 1:
        cos_jk = 1
    if cos_jk < -1:
        cos_jk = -1
    return np.array([cos_ij, cos_ik, cos_jk])

def three_body(kappas, distances, bo):
    cosines = calc_cosines(*distances)
    value = np.prod(kappas) * np.prod(2 * np.abs(bo)) * np.prod(cosines)
    return value

def three_energy_decomp(three_two_decomp, three_body_value, two_bodies, pairs):
    
    # two body gross contributions
    two_ij = two_bodies[pairs[0]]
    two_ik = two_bodies[pairs[1]]
    two_jk = two_bodies[pairs[2]]
    sum_two = two_ij + two_ik + two_jk
    
    # three-body contributions decomposed into two bodies
    three_two_decomp[pairs[0]] += three_body_value * (two_ij / sum_two)
    three_two_decomp[pairs[1]] += three_body_value * (two_ik / sum_two)
    three_two_decomp[pairs[2]] += three_body_value * (two_jk / sum_two)
    return three_two_decomp

# MAIN ZPVE EQUATION
def zpe(atoms, bo, distance_matrix, parameter_folder):
    """Calculate zpve energy and zpve bond 
    
       Parameters
       ----------
       atoms: :obj:'numpy.ndarray'
           array showing the atoms
       bo: :obj:'numpy.ndarray'
           Mulliken bond order
           
       Returns
       -------
       zpve: :obj:'numpy.float64'
           the total zpve in kcal/mol
       bond_zpve: :obj:'numpy.ndarray'
           the zpve bond energies in kcal/mol
       """
    size = atoms.shape[0]
    harmonic = np.zeros((size,size), dtype = np.float64) # harmonic contributions
    anharmonic = np.zeros((size,size), dtype = np.float64) # short-range anharmonics
    three_two_decomp = np.zeros((size,size), dtype = np.float64) # three-body
    
    # two-body contributions
    for l in range(1,size):
        for n in range(l):
            # important properties
            index = (l,n)
            pair_atoms = np.array([atoms[l], atoms[n]])
            bo_pair = bo[l][n]
            distance = distance_matrix[l][n]
            
            anti_corr = conditional_statements(pair_atoms, bo_pair, size)
            
            # fitting parameters
            beta, pre_exp, zeta, R_param = param.zpe_pair(pair_atoms, parameter_folder, anti_corr)
            

            # extended Hückel covalent bonding for sigma and pi bonding 
            harmonic[l][n] = E_harm(bo_pair, beta)
            anharmonic[l][n] = E_anharm(distance, pre_exp, zeta, R_param)
                       
    # Total Pairwise Interactions
    two_bodies = harmonic + anharmonic
    
    # three-body contributions
    three_energy = 0
    n = 0
    bonds = np.array(['C~C','N~N','O~O','B~B','B~N',
                      'N~B','O~B','B~O','B~C','C~B',
                      'C~O','O~C','C~N','N~C','N~O',
                      'O~N'])
    for i in range(2,size):
        for j in range(1,i):
            for k in range(j):
                    bond_ij = f'{atoms[i]}~{atoms[j]}'
                    bond_ik = f'{atoms[i]}~{atoms[k]}'
                    bond_jk = f'{atoms[j]}~{atoms[k]}'
                    three_bonds = np.array([bond_ij, bond_ik, bond_jk])
                    if np.prod(np.isin(three_bonds,bonds)) == 1:
                        n += 1
                        pairs = [(i,j),(i,k),(j,k)]
                        distances = np.array([distance_matrix[l] for l in pairs])
                        pair_atoms = np.array([[atoms[l],atoms[z]] for l, z in pairs])
                        bo_pair = np.array([bo[l] for l in pairs])
                        anti_corr = np.array([conditional_statements(pair_atoms[i], bo_pair[i], size) for i in range(3)])
                        kappas = np.array([param.zpe_pair(pair_atoms[l], parameter_folder, anti_corr[l], two_body = False) for l in range(3)])
                        three_body_value = three_body(kappas, distances, bo_pair)
                        three_energy += three_body_value
                        three_two_decomp = three_energy_decomp(three_two_decomp, three_body_value, two_bodies, pairs)
                    else:
                        continue
                        
    zpve = np.sum(two_bodies) + three_energy
    return (zpve, two_bodies, three_two_decomp) 

def zpebop2_bond_energy(zpve, two_body, three_body):
    """Calculate the individual vibrational bond energy contributions
    
    Parameters
    ----------
    zpve: :obj:'numpy.float64'
        zero-point vibrational energy at 0K in a molecule
    two_body: :obj:'numpy.ndarray'
        the two-body vibrational bond energies at 0 K
    three_body: :obj:'numpy.ndarray'
        the three-body vibrational bond energies at 0 K 
        (decomposed to two-body contributions)
        
    Returns
    -------
    E_gross: :obj:'numpy.ndarray' 
        gross vibrational bond energies (i.e., E_gross = E_Huckel,vib + short-range anharmonic)
    E_net: :obj:'numpy.ndarray'
        net vibrational bond energies (i.e., E_net = E_gross + three_body)
    CompositeTable: :obj:'numpy.ndarray'
        matrix showing gross vibrational bond energies(upper diagonal elements) 
        and net vibrational bond energies (lower diagonal elements)
    """
    
    size = two_body.shape[0]
    E_gross = two_body
    E_net = two_body + three_body
    CompositeTable = np.zeros((size,size), dtype = np.float64)
    
    for i in range(1,size):
        for j in range(i):
            CompositeTable[j][i] = E_gross[i][j] # gross energies for 
                                                 # the upper diagonal matrices
            CompositeTable[i][j] = E_net[i][j] # net bond energies for
                                               # the lower diagonal matrices
    
    return (E_gross, E_net, CompositeTable)
