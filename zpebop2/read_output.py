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


"""Script that reads the open-restricted shell Hartree-Fock/CBSB3 output file from Gaussian16
   read_atoms :: reads the atoms from the Hartree/Fock output file
   read_coordinates :: reads the coordinates present in the output file
   read_MBS_DensityMatrix_Pop :: reads the density matrix (alpha and beta spin states)
   and the population matrix
   read_Mulliken :: reads the Mulliken MBS bond order matrix  
   read_gross_orbitals :: read all the gross 2s orbitals for hybridization calculations 
   read_entire_output :: read all of information from the output file (i.e., all subroutines mentioned 
   above are executed)
   """

import numpy as np

def read_atoms(vertical_position):
    """Read the atoms in the Hartree-Fock File (this is before the distance matrix)
    
       Parameter 
       ----------
       vertical_position: :obj:'str' or 'list' of 'str' 
           The list of strings present in the ROHF file 
        
       Return
       -------
       nAtoms: :obj: 'np.ndarray'
           Atoms present in the molecule
       """
    
    Elements = np.array(['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F',
                         'Ne','Na', 'Mg', 'Al','Si', 'P', 'S', 'Cl', 'Ar'])
    nAtoms = np.array([])
    for l1 in vertical_position:
        if l1.startswith(' \n') or l1.startswith('       Variables:'):
            break          
        else:
            Atoms = l1[0:3].replace('\n','')          
            Atomic = Atoms.replace(' ','')           
            if Atomic in Elements:   
                nAtoms = np.concatenate((nAtoms,np.array([Atomic])))                
            else:            
                continue     
    return nAtoms

def read_coordinates(vertical_position):
    """Read the 'Standard Orientation' xyz geometries
    
       Parameter
       ---------
       vertical_position: :obj:'str' or 'list' of 'str' 
           The list of strings present in the ROHF file 
           
       Return
       ------
       XYZ: :obj:'np.ndarray'
           The standard orientation xyz geometries
           
       """
    
    # Read the XYZ file
    n = 0
    for l in vertical_position:
        if l.startswith(' --------'):
            break         
        else:       
            X, Y, Z = l[34::].split()
            if n == 0: 
                XYZ = np.array([np.array([np.float64(X),np.float64(Y),np.float64(Z)])])
                n = 1
            else:
                XYZ = np.concatenate((XYZ,np.array([np.array([np.float64(X),np.float64(Y),np.float64(Z)])])))
    return XYZ

def read_MBS_DensityMatrix_Pop(vertical_position, nAtoms, Alpha=False, Beta=False, Pop=False, TotalOrb=None, columns = 0):
    """Read the alpha and beta CiCj and Mulliken population matrices
       Note: this prototype is reading under the following title of the ROHF output file:
       'Alpha MBS Density Matrix', 'Beta MBS Density Matrix', and 'Full MBS Mulliken population analysis'
    
       Parameter
       ---------
       vertical_position: :obj:'str' or 'list' of 'str' 
           The list of strings present in the ROHF file 
       nAtoms: :obj: 'np.ndarray'
           Atoms present in the molecule
       Alpha: :obj:'bool', optional
           Get CiCj alpha matrix condensed to orbitals?
       Beta: obj:'bool', optional
           Get CiCj beta matrix condensed to orbitals?
       Pop: :obj:'bool', optional
           Get MBS Mulliken population matrix condensed to orbitals? 
       TotalOrb: :obj:'int', optional
           Total number of orbitals in the molecule
       columns: :obj:'np.ndarray', optional
           Correct number of columns (important for Beta and Pop calculations)
       
       Returns
       -------
       MBS: :obj:'np.ndarray' 
           Return the alpha and beta CiCj matrices and Mulliken population analysis 
       NISTBF: :obj:'np.ndarray'
           Return the indexes where the orbitals are located for each atom (this is done once after returning CiCj Alpha)
       TotalOrb: :obj:'int'
           Return the total number of orbitals within the molecule (this is done once after returning CiCj Alpha)
       opt2: :obj:'np.ndarray'
           Return the array with the number # the number in the first column after reaching the TotalOrb row number 
           (Gaussian16 will create another row not part of matrix after this number) 
       """
    
    v = 0 
    n = 0 
    indexes = np.array([]) # dummy variable to read the indexes
    elementsTot = np.array([]) # dummy variabe for the total number of arrays
    if Alpha == True: # this is done only once
        im = 0
        TotalOrb = 0 # dummy variable for the total number of orbitals present in the molecule
        NISTBF = np.array([]) # dummy variable for elements not part of the Density Matrix
        for l in nAtoms:
            NISTBF = np.concatenate((NISTBF,np.array([im])))
            im += 1
            if l == 'H' or l == 'He':
                TotalOrb += 1 # for 1s orbital
            elif l in np.array(['Li','Be','B','C','N','O','F']):
                im += 4 # 2s, 2px-2py-2pz orbitals
                TotalOrb += 5
            else:
                im += 13 # 2s, 2px-2py-2pz, 3s, 3px-3py-3pz, 4d0-4d1-4d-1-4d2-4d-2 orbitals
                TotalOrb += 14
                
        columns = np.array(5 * np.arange(1,TotalOrb, dtype=np.int) + 1, dtype = np.str) # the number in the first column after
                                                                                           # reaching the TotalOrb row number 
                                                                                           # (Gaussian will create another row after this number)
            
    Matrix = np.array([]) # dummy variable for the matrix
    # last title the program needs to read to stop reading the density matrix
    if Alpha == True:
        end_title = '     Beta  MBS Density Matrix:'
    elif Beta == True:
            
        end_title = '    Full MBS Mulliken population analysis:'
            
    elif Pop == True:
        end_title = '     MBS Gross orbital populations:' 
    for t in vertical_position:
        if t.startswith(end_title):
            break
        else:
            rows = np.zeros(5) # the number of constant rows in the matrix of CiCj and Pop (5 in Gaussian16)
            Elements = np.array(t[23::].split(),dtype=np.float64)
            elementsTot = np.concatenate((elementsTot,np.array([Elements.shape[0]])))
            rows[:Elements.shape[0]] = Elements
            if n == 0:
                Matrix = np.array([rows]) 
                n = 1
            else:
                Matrix = np.concatenate((Matrix,np.array([rows])))
            v += 1 
            if t[23:28].replace(' ','') in columns:    
                indexes = np.concatenate((indexes,np.array([v - 1]))).astype(np.int)
                    
    # Correct the CiCj Matrix
    size = indexes.shape[0] # total extra length not part of CiCj or Pop Matrix
    length = Matrix.shape[0] - size # what is the correct length of the CiCjMatrix or Pop Matrix
    if length < 5:
        AllParts = np.zeros((length,length),dtype=np.float64)
        for l in range(length): 
            AllParts[l,:elementsTot[l].astype(np.int)] = Matrix[l,:elementsTot[l].astype(np.int)]
    elif length == 5:
        AllParts = Matrix
    else:
        AllParts = Matrix[:indexes[0]]
        for l in range(len(indexes)):
            if len(indexes[l:]) < 2:
                Parts = Matrix[indexes[l] + 1:]
            else:
                Parts = Matrix[indexes[l] + 1:indexes[l+1]]
            sizemissing = TotalOrb - Parts.shape[0]
            zeroes = np.zeros((sizemissing,5))
            Parts = np.concatenate((zeroes,Parts))
            AllParts = np.concatenate((AllParts,Parts),axis=1)       
            
    # correct matrix is nxn where n is the total number of orbitals array present in the molecule   
    if Alpha == True:
        return (AllParts[:TotalOrb,:TotalOrb], NISTBF.astype(np.int), TotalOrb, columns)
    else:  
        return AllParts[:TotalOrb,:TotalOrb]
    
def read_Mulliken(vertical_position, nAtoms, TotalOrbs):
    """Read the Mulliken bond orders and organized the elements to a square matrix
    
       Note: this code reads the following 'MBS Condensed to atoms (all electrons)'
    
       Parameters
       ----------
       vertical_position: :obj:'str' or 'list' of 'str' 
           The list of strings present in the ROHF file 
       nAtoms: :obj: 'np.ndarray'
           Atoms present in the molecule
       TotalOrb: :obj:'int', optional
           Total number of orbitals in the molecule    
       
       Returns
       -------
       Mulliken_Matrix: :obj:'np.darray'
           Mulliken matrix condensed to atoms
       """
    
    n = 0
    v = 0
    indexes = np.array([])
    column2 = np.array(6 * np.arange(1,TotalOrbs) + 1, dtype=np.str)
    
    # read the elements of the Mulliken Bond Orders
    for l in vertical_position:
        if l.startswith('          MBS Atomic-Atomic Spin Densities.'):
            break
        else:
            if n == 0:
                Mulliken = np.array([np.array(l[12::].split(), dtype = np.float64)])
                n = 1
            else: 
                splitcell = np.array(l[12::].split(), dtype = np.float64)
                if splitcell.shape[0] != Mulliken.shape[1]:
                    diff = np.abs(splitcell.shape[0] - Mulliken.shape[1]).astype(np.int)
                    extra = np.array([np.concatenate((splitcell, np.zeros(diff)))])
                    Mulliken = np.concatenate((Mulliken, extra))
                else:
                    Mulliken = np.concatenate((Mulliken,np.array([splitcell])))
            v += 1
            if l[11:20].replace(' ','').replace('\n','') in column2:
                indexes = np.concatenate((indexes,np.array([v-1]))).astype(np.int)
        
    # Organized the Mulliken Matrices
    size_indexes = indexes.shape[0] # size of the indexes
    size_Atoms = nAtoms.shape[0] # size of the atom array
    if size_indexes != 0:
        Mulliken_Matrix = np.array(Mulliken[:indexes[0]])
        for t in range(size_indexes):
            Parts = np.array(Mulliken[indexes[t] + 1:indexes[t] + size_Atoms + 1])
            Mulliken_Matrix = np.concatenate((Mulliken_Matrix,Parts),axis=1)
        return Mulliken_Matrix[:size_Atoms,:size_Atoms]   
    else:
        return Mulliken
    
def read_gross_orbitals(vertical_position): 
    """Read the gross orbitals
       Note: this code reads the following 'MBS Gross orbital populations'
       
       Parameters
       ----------
       vertical_position: :obj:'str' or 'list' of 'str' 
           The list of strings present in the ROHF file 
       nAtoms: :obj: 'np.ndarray'
           Atoms present in the molecule
           
       Returns
       -------
       Occ2s: :obj:'np.ndarray' or 'np.float64'
           Molecular 2s occupations of atoms
       """
    
    AllOcc2s = []
    for l in vertical_position:
        if l[12:15].startswith('2S'):
            AllOcc2s += np.float(l[15:31].replace(' ','')),
        elif l.startswith('          MBS Condensed to atoms (all electrons):'):
            break
            
    AllOcc2s = np.array(AllOcc2s)
    if (AllOcc2s.size == 1) or (AllOcc2s.size > 1):
        Occ2s = AllOcc2s
    else:
        Occ2s = 0
    
    return Occ2s
    
def fix_gross_orbital(Occ2s_test, nAtoms):
    """Make the gross orbital arrays include orbital populations for 2s and 2p for H and He.
       
       Parameters
       ----------
       Occ2s_test: :obj:'np.ndarray' or 'np.float64'
           Molecular 2s occupations of atoms
       nAtoms: :obj: 'np.ndarray'
           Atoms present in the molecule
           
       Returns
       -------
       Occ2s_test: :obj:'np.ndarray'
           Molecular 2s occupations for all atoms
    """
    
    size = nAtoms.shape[0]
    gross2s = np.zeros(size, dtype = np.float64)
    if type(Occ2s_test) == np.ndarray:
        ns = 0
        for l in range(size):
            if nAtoms[l] in np.array(['H','He']):
                continue
            else:
                gross2s[l] = Occ2s_test[ns]
                ns += 1
    return gross2s

def MBS_Charges(vertical_position):
    """Read the charges from the first row of 'MBS Mulliken charges and 
       spin densities with hydrogens summed into heavy atoms:'
       
       Parameters
       ----------
       vertical_position: :obj:'str' or 'list' of 'str' 
           The list of strings present in the ROHF file 
           
       Returns
       -------
       Charges: :obj:'np.ndarray'
           Charges of the atoms within a bond
    """
    
    Charges = np.array([])
    for l in vertical_position:
        if l.startswith(' Sum of MBS Mulliken charges ='):
            break
        else:
            charg = l[11:22].replace(' ','')
            charg = np.float64(charg)
            Charges = np.concatenate((Charges,np.array([charg])))
    return Charges
    
    
def read_entire_output(FILE):
    """Read the entire Hartree-Fock output file containing all essential bond orders
       necessary for the ZPEBOP calculations
       
       Parameters
       ----------
       FILE: :obj:'str'
           Name of the file/path
       
       Returns
       -------
       nAtoms: :obj:'np.ndarray'
           Atoms present in the molecule
       XYZ: :obj:'np.ndarray'
           The standard orientation xyz geometries
       CiCjAlpha: :obj:'np.ndarray'
           The alpha CiCj matrices
       CiCjBeta: :obj:'np.ndarray'
           The beta CiCj matrices
       PopMatrix: :obj:'np.ndarray'
           Mulliken population analysis condensed to orbitals
       NISTBF: :obj:'np.ndarray'
           Return the indexes where the orbitals are located for each atom 
           (this is done once after returning CiCj Alpha)
       Occ2s: :obj:'np.ndarray' or np.float
           Molecular 2s occupations of atoms
       Mulliken: :obj:'np.ndarray'
           Mulliken matrix condensed to atoms 
       Charges: :obj:'np.ndarray'
           Charges of the atoms within a bond
    """
    
    o = 0
    with open(FILE, mode='r') as ROHF:
        p = ROHF.readlines()[91:]
        for n in p:
            o += 1
            if n.startswith(' Charge ='):
                nAtoms = read_atoms(p[o::])
            elif n.startswith('                         Standard orientation:'):
                XYZ = read_coordinates(p[o+4::])
            elif n.startswith('     Alpha  MBS Density Matrix:'):    
                CiCjAlpha, NISTBF, OrbN, Columns = read_MBS_DensityMatrix_Pop(p[o+1::],nAtoms,Alpha=True)
            elif n.startswith('     Beta  MBS Density Matrix:'):    
                CiCjBeta = read_MBS_DensityMatrix_Pop(p[o+1::],nAtoms,Beta=True,TotalOrb=OrbN, columns=Columns)
            elif n.startswith('    Full MBS Mulliken population analysis:'):
                PopMatrix = read_MBS_DensityMatrix_Pop(p[o+1::],nAtoms,Pop=True,TotalOrb=OrbN, columns=Columns)
            elif n.startswith('     MBS Gross orbital populations:'):
                Occ2s_test = read_gross_orbitals(p[o::])
                Occ2s = fix_gross_orbital(Occ2s_test, nAtoms)
            elif n.startswith('          MBS Condensed to atoms (all electrons):'):
                Mulliken = read_Mulliken(p[o+1::], nAtoms, OrbN)
            elif n.startswith(' MBS Mulliken charges and spin densities:'):
                Charges = MBS_Charges(p[o+1::])
            elif n.startswith(' Sum of MBS Mulliken charges ='):
                break
                
    ROHF.close()  
    return (nAtoms, XYZ, CiCjAlpha, CiCjBeta, PopMatrix, NISTBF, Occ2s, Mulliken, Charges)