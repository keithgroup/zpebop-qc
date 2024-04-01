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

"""Main subroutine that does all calculation
   zpve_results :: compute the zpve and zpve bond energies in Hartrees
   bond_E :: compute the bond energy tables
   sort_bondE:: sort the vibrational bond energies
   """

import numpy as np
from . import zpebop1_equation as zpve_eq 
from . import read_output as ro

class ZPVE: # zpve class
    
    def __init__(self, name):
        """Give the name of the B3LYP output file to get the bond energy data.
        
        Parameters
        ----------
        name: :obj:'str'
            Name of the B3LYP/CBSB3 output file
        """
        
        self.name = name # name(or path+name) of the Gaussian output file
        nAtoms, XYZ, CiCjAlpha, CiCjBeta, PopMatrix, NISTBF, Occ2s, Mulliken, charges = ro.read_entire_output(name) # get all data 
                                                                                                                    # from the output file
        self.mol = nAtoms # array containing the elements within the molecule
        self.mulliken = Mulliken # Mulliken MBS bond orders condensed to atoms
        return None
    
    def zpve_results(self, units = 'Hartrees'):
        """Compute zpve energy in Hartrees
        
        Output
        ------
        self.zpve: :obj:'numpy.float64'
            total zero-point vibrational energy at 0 K in a molecule
        self.zpve_bonds: :obj:'np.ndarray'
            vibrational bond energies at 0 K in a molecule
        units: :obj:'str'
            units of the ZPE and vibrational bond energies. Options include:
            'kcal/mol' - for kcal/mol units 
            'Hartrees', 'hartrees', 'Eh', 'au', 'AU' - for Hartree units
            
            Default units are Hartrees. 
        """
        
        self.zpve, self.zpve_bonds = zpve_eq.zpve(self.mol, self.mulliken)
        if units in np.array(['Hartrees','hartrees','Eh','au','AU']):
            return (self.zpve, self.zpve_bonds)
        elif units in np.array(['kcal/mol']):
            conversion = 627.5096 # 1 Eh = 627.5097 kcal/mol
            self.zpve *= conversion
            self.zpve_bonds *= conversion
            return (self.zpve, self.zpve_bonds)
    
    def bond_E(self, NetBond = True, GrossBond = True, Composite = True):
        """Compute vibrational bond energies in kcal/mol.All bond energies are printed by default.
           User may select which energies will be returned by selecting 'True'.
           
           New users are advise to use the '.keys()' method on the output variables to check 
           the name of the keys in the dictionaries.
           
           Parameters
           ----------
           NetBond: :obj:'bool',optional
               Generate the bond energy to include the extended vibrational Hückel model.  
               Returns net covalent bond energies (Enet) if 'NetBond = True'.
           GrossBond: :obj:'bool', optional
               Generate the bond energy to include repulsion corrections only 
               Returns covalent bond energies (Ecov) if 'GrossBond = True'.
           Composite: :obj:'bool', optional
               Return the composite table (CompositeTable) containing the net bond energies (lower diagonal elements), 
               and the gross bond energies (upper diagonal elements)
               
               
          Returns
          -------
           DictionaryTotal: :obj:'dict'
               Dictionary containing Enet (key: 'NetBond'), Ecov (key: 'GrossBond'), 
               and CompositeTable (key: 'Composite')
           """
        AllTotalEnergies = []
        keysTotalE = []
        data = zpve_eq.zpebop1_bond_energy(self.zpve, self.zpve_bonds) # get all data
        self.E_gross = data[0]
        self.E_net = data[1]
        self.CompositeTable = data[2]
        
        # store all of the data in a list 
        if GrossBond == True:
            AllTotalEnergies += self.E_gross, 
            keysTotalE += 'GrossBond',
        if NetBond == True:
            AllTotalEnergies += self.E_net,
            keysTotalE += 'NetBond',
        if Composite == True:
            AllTotalEnergies += self.CompositeTable, 
            keysTotalE += 'CompositeTable',
            
        # get the nmaes of the keys, and bring arrays to its respective keys
        DictionaryTotal = {key: value for key, value in zip(keysTotalE, AllTotalEnergies)}
        
        return DictionaryTotal
    
    def sort_bondE(self,bond_energies,rel= False, with_number= True):
        """Sort the relative bond energies from strongest to weakest in energy.
           User can request not to have absolute by 'rel= False'.
           Also, user can request whether they would like relative or absolute anti-bonding energies.
           
           Parameters
           ----------
           bond_energies: :obj:'np.ndarray'
               Any bond energy array from bond_E(). This subroutine will not work for composite tables. 
           rel: :obj:'bool', optional
               Return relative bond energies  
           with_number: :obj:'bool', optional
               Put the atom number to distinguish the bond energy
           
           Returns
           -------
           sort_bonds: :obj:'dict' or 'np.ndarray'
                The bonding (key: 'bonding') and/or antibonding(key:'antibonding') identity in the molecule
                (i.e., gives the bond between two atoms with/without the atom number shown in the ROHF input)
           sort_BE: :obj:'dict' or np.darray
                Values of the bonding (key: 'bonding') and/or anti-bonding (key:'antibonding') arrays
                
           """
        
        sort_bonds = []
        sort_be = []
        size = bond_energies.shape[0]
        
        if with_number == True: # user wants to distinguish the identity of the bond
            position = np.array(np.arange(1,size + 1), dtype=np.str) # indicate the position of the atoms 
            newAtoms = np.core.defchararray.add(self.mol, position) 
        else: # user does not want to distinguish bons
            newAtoms = self.mol
        for l in range(1, size): 
            for n in range(l):
                if bond_energies[l][n] < 0.01:
                    continue
                else:
                    sort_bonds += f'{newAtoms[n]}-{newAtoms[l]}',
                    sort_be += bond_energies[l][n],
        
        # create numpy array
        sort_bonds = np.array(sort_bonds)
        sort_be = np.array(sort_be)
        
        # Sort the bond energies and bonding index
        n = np.argsort(sort_be)
        sort_be = np.sort(sort_be)
        sort_bonds = sort_bonds[n]
        
        # Sort the bond type
        if rel == True:
            sort_be = sort_be - sort_be[0] 
            return (sort_bonds, sort_be)
        else:
            return (sort_bonds, sort_be)
            
