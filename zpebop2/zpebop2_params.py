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

"""Parameters for the ZPEBOP-2 code (version 1.0.0)
   zpve_pair :: ZPEBOP atom-pair parameters (i.e., beta)
   """

import numpy as np
import json
from os import path

def zpe_pair(pair_atoms, parameter_folder_path, anti_corr, two_body = True):
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
    bonds1 = f'{pair_atoms[0]}~{pair_atoms[1]}'
    bonds2 = f'{pair_atoms[1]}~{pair_atoms[0]}'
    if path.exists(f'{parameter_folder_path}{bonds1}/{bonds1}_param_opt.json') == True:
        file = f'{parameter_folder_path}{bonds1}/{bonds1}_param_opt.json'
    elif path.exists(f'{parameter_folder_path}{bonds2}/{bonds2}_param_opt.json') == True:
        file = f'{parameter_folder_path}{bonds2}/{bonds2}_param_opt.json'
    with open(file, "r") as openfile:
        parameter = json.load(openfile)
    
    if two_body == True:
        if anti_corr == False:
            beta_val = parameter['beta_bond']
        else:
            beta_val = parameter['beta_anti']
        pre_exp = parameter['pre_exp']
        zeta = parameter['zeta']
        R_param = parameter['R_param']
        return (beta_val, pre_exp, zeta, R_param)
    
    else:
        if anti_corr == False:
            kappa = parameter['kappa_bond']
        else:
            kappa = parameter['kappa_anti']
        return kappa
