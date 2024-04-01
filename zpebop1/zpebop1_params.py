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

"""Parameters for the ZPEBOP-1 code (version 1.0.0)
   zpve_pair :: ZPEBOP atom-pair parameters (i.e., beta)
   """

import numpy as np
import json

def zpve_pair(pair_atoms, anti_corr):
    """Fitted atom-pair parameters used in the ZPEBOP equation 
    
       Parameters
       ----------
       pair_atoms: :obj:'np.ndarray'
           Name of the two-pair atoms 
       
       Returns 
       -------
       beta_val: :obj:'np.float'
           Fixed parameter used compute the extended-Hückel bond energy for sigma bonding  
       """

    AtomN = {'H': 0,
             'He': 1,
             'Li': 2,
             'Be': 3,
             'B': 4,
             'C': 5,
             'N': 6,
             'O': 7,
             'F': 8,
             'Ne':9,
             'Na':10,
             'Mg':11,
             'Al':12,
             'Si':13,
             'P':14,
             'S':15,
             'Cl':16,
             'Ar':17}
    
    args = np.argsort(np.array([AtomN[pair_atoms[0]], AtomN[pair_atoms[1]]]))
    atoms = [pair_atoms[args[0]], pair_atoms[args[1]]] 
    
    if anti_corr == False: # bonding
        if (AtomN[atoms[0]] <= 8) and (AtomN[atoms[1]] <= 8):
               #               H       He      Li      Be       B        C        N        O         F
            beta =  {'H':[  0.01257,   ' ',    ' ',     ' ',     ' ',     ' ',     ' ',     ' ',     ' '],
                     'He':[   'NaN', 'NaN',    ' ',     ' ',     ' ',     ' ',     ' ',     ' ',     ' '],
                     'Li':[ 0.00426, 'NaN',0.00112,     ' ',     ' ',     ' ',     ' ',     ' ',     ' '],
                     'Be':[ 0.00821, 'NaN',0.00133, 0.00250,     ' ',     ' ',     ' ',     ' ',     ' '],
                     'B':[  0.00960, 'NaN',0.00171, 0.00228, 0.00378,     ' ',     ' ',     ' ',     ' '],
                     'C':[  0.01080, 'NaN',0.00253, 0.00472, 0.00398, 0.00400,     ' ',     ' ',     ' '],
                     'N':[  0.01328, 'NaN',0.00217, 0.00455, 0.00264, 0.00373, 0.00401,     ' ',     ' '],
                     'O':[  0.01690, 'NaN',0.00103, 0.00222, 0.00140, 0.00474, 0.00718, 0.00642,     ' '],
                     'F':[  0.02179, 'NaN',0.00523, 0.00603, 0.00647, 0.00648, 0.00910, 0.00781, 0.00955]}
            
            beta_val = beta[atoms[1]][AtomN[atoms[0]]]
        
        elif (AtomN[atoms[0]] <= 8) and (AtomN[atoms[1]] > 8):
               #               Ne      Na      Mg      Al       Si        P        S        Cl       Ar
            beta =  {'H':[    'NaN',0.00792,0.00879, 0.00888, 0.00965, 0.01172, 0.01291, 0.01344,   'NaN'],
                     'He':[   'NaN', 'NaN',   'NaN',   'NaN',   'NaN',   'NaN',   'NaN',   'NaN',   'NaN'],
                     'Li':[   'NaN',0.01646,0.00304, 0.00119, 0.00129, 0.00193, 0.00180, 0.00370,   'NaN'],
                     'Be':[   'NaN',0.00746,0.00273, 0.00225, 0.00195, 0.00235, 0.00252, 0.00438,   'NaN'],
                     'B':[    'NaN',0.00455,0.00309, 0.00221, 0.00277, 0.00100, 0.00302, 0.00378,   'NaN'],
                     'C':[    'NaN',0.00272,0.00853, 0.00415, 0.00276, 0.00019, 0.00333, 0.00378,   'NaN'],
                     'N':[    'NaN',0.00326,0.00244, 0.00227, 0.00179, 0.00099, 0.00374, 0.00534,   'NaN'],
                     'O':[    'NaN',0.00021,0.00197, 0.00309, 0.00269, 0.00310, 0.00346, 0.00299,   'NaN'],
                     'F':[    'NaN',0.00250,0.00323, 0.00431, 0.00438, 0.00481, 0.00451, 0.00702,   'NaN']}
            
            beta_val = beta[atoms[0]][AtomN[atoms[1]] - 9]
        
        elif (AtomN[atoms[0]] > 8) and (AtomN[atoms[1]]  > 8):
               #                 Ne     Na      Mg      Al       Si        P        S        Cl       Ar
            beta =  {'Ne':[    'NaN',    ' ',    ' ',     ' ',     ' ',     ' ',     ' ',     ' ',     ' '],
                     'Na':[    'NaN',0.00231,    ' ',     ' ',     ' ',     ' ',     ' ',     ' ',     ' '],
                     'Mg':[    'NaN',0.00186,0.00100,     ' ',     ' ',     ' ',     ' ',     ' ',     ' '],
                     'Al':[    'NaN',0.00235,0.00248, 0.00195,     ' ',     ' ',     ' ',     ' ',     ' '],
                     'Si':[    'NaN',0.00177,0.00230, 0.00170, 0.00188,     ' ',     ' ',     ' ',     ' '],
                     'P':[     'NaN',0.00148,0.00193, 0.00188, 0.00153, 0.00228,     ' ',     ' ',     ' '],
                     'S':[     'NaN',0.00117,0.00176, 0.00240, 0.00183, 0.00200, 0.00277,     ' ',     ' '],
                     'Cl':[    'NaN',0.00195,0.00171, 0.00325, 0.00306, 0.00305, 0.00339, 0.00352,     ' '],
                     'Ar':[    'NaN', 'NaN',  'NaN',   'NaN',   'NaN',   'NaN',   'NaN',   'NaN',   'NaN']}
            
            beta_val = beta[atoms[1]][AtomN[atoms[0]] - 9]
        
    elif anti_corr == True: # anti-bonding
        if (AtomN[pair_atoms[0]] <= 8) and (AtomN[pair_atoms[1]] <= 8):
            #                  H     He       Li       Be        B        C        N        O        F
            beta =  {'H':[ 0.04186,   ' ',     ' ',     ' ',     ' ',     ' ',     ' ',     ' ',     ' '],
                     'He':[  'NaN', 'NaN',     ' ',     ' ',     ' ',     ' ',     ' ',     ' ',     ' '],
                     'Li':[0.00746, 'NaN', 0.00672,     ' ',     ' ',     ' ',     ' ',     ' ',     ' '],
                     'Be':[0.01993, 'NaN', 0.00410, 0.00221,     ' ',     ' ',     ' ',     ' ',     ' '],
                     'B':[ 0.04080, 'NaN', 0.00909, 0.00543, 0.01641,     ' ',     ' ',     ' ',     ' '],
                     'C':[ 0.03281, 'NaN', 0.00000, 0.00000, 0.00000, 0.00674,     ' ',     ' ',     ' '],
                     'N':[ 0.04483, 'NaN', 0.00000, 0.00000,-0.03488, 0.00962, 0.03265,     ' ',     ' '],
                     'O':[ 0.08411, 'NaN', 0.01410, 0.00000, 0.06328, 0.01121, 0.00838, 0.02517,     ' '],
                     'F':[ 0.05515, 'NaN',   'NaN', 0.00873, 0.01783, 0.01865, 0.02160, 0.03610, 0.02597]}
            
            beta_val = -beta[atoms[1]][AtomN[atoms[0]]]
        
        elif (AtomN[atoms[0]] <= 8) and (AtomN[atoms[1]] > 8):
               #               Ne      Na      Mg      Al       Si        P        S        Cl       Ar
            beta =  {'H':[    'NaN',0.00000,0.00400, 0.04426, 0.03679, 0.07283, 0.05013, 0.03870,   'NaN'],
                     'He':[   'NaN', 'NaN',   'NaN',   'NaN',   'NaN',   'NaN',   'NaN',   'NaN',   'NaN'],
                     'Li':[   'NaN', 'NaN',   'NaN',   'NaN', 0.03169,   'NaN',   'NaN',   'NaN',   'NaN'],
                     'Be':[   'NaN', 'NaN',   'NaN',   'NaN', 0.01256, 0.00091, 0.00000, 0.00024,   'NaN'],
                     'B':[    'NaN', 'NaN', 0.00049,   'NaN',-0.00084, 0.00000, 0.01472, 0.00403,   'NaN'],
                     'C':[    'NaN',0.0000, 0.00000, 0.00000, 0.00983, 0.00200, 0.01292, 0.00697,   'NaN'],
                     'N':[    'NaN', 'NaN', 0.00248, 0.00000, 0.02524, 0.07340, 0.03765, 0.00000,   'NaN'],
                     'O':[    'NaN', 'NaN',   'NaN',   'NaN', 0.00000, 0.01632, 0.02133, 0.02065,   'NaN'],
                     'F':[    'NaN', 'NaN',   'NaN',   'NaN', 0.01252, 0.00000, 0.02435, 0.00263,   'NaN']}
            
            beta_val = -beta[atoms[0]][AtomN[atoms[1]] - 9]
        
        elif (AtomN[atoms[0]] > 8) and (AtomN[atoms[1]]  > 8):
               #                 Ne     Na      Mg      Al       Si        P        S        Cl       Ar
            beta =  {'Ne':[    'NaN',    ' ',    ' ',     ' ',     ' ',     ' ',     ' ',     ' ',     ' '],
                     'Na':[    'NaN',0.00000,    ' ',     ' ',     ' ',     ' ',     ' ',     ' ',     ' '],
                     'Mg':[    'NaN',0.00535,  'NaN',     ' ',     ' ',     ' ',     ' ',     ' ',     ' '],
                     'Al':[    'NaN',  'NaN',  'NaN',0.001098,     ' ',     ' ',     ' ',     ' ',     ' '],
                     'Si':[    'NaN',  'NaN',  'NaN',   'NaN', 0.00085,     ' ',     ' ',     ' ',     ' '],
                     'P':[     'NaN',  'NaN',0.00098,   'NaN', 0.00994, 0.03367,     ' ',     ' ',     ' '],
                     'S':[     'NaN',  'NaN',  'NaN', 0.00000, 0.01105, 0.01771, 0.02037,     ' ',     ' '],
                     'Cl':[    'NaN',  'NaN',  'NaN',   'NaN', 0.02367, 0.00905, 0.00000, 0.00491,     ' '],
                     'Ar':[    'NaN',  'NaN',  'NaN',   'NaN',   'NaN',   'NaN',   'NaN',   'NaN',   'NaN']}
    
            beta_val = -beta[atoms[1]][AtomN[atoms[0]] - 9]
    return beta_val
