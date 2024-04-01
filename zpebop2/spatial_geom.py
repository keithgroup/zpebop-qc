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

"""Compute the distance matrix
   Spatial_Properties :: Compute the distance matrix
   """

import numpy as np

def spatial_properties(XYZ):
    """Compute the distance matrix and the trig. projection functions
    
       Parameters
       ----------
       XYZ: obj:'np.ndarray'
           Standard orientation cartesian coordinates.
       
       Output
       ------
       distance_matrix: obj:'np.ndarray'
           Calculated distance matrix (should agree with Gaussian16 output).
    """
    
    length = XYZ.shape[0] 
    distance_matrix = np.zeros((length,length), dtype=np.float64)
        
    for i in range(1,length):
        for j in range(i):
            DeltaX = XYZ[j][0] - XYZ[i][0]
            DeltaY = XYZ[j][1] - XYZ[i][1]
            DeltaZ = XYZ[j][2] - XYZ[i][2]
            distance_matrix[i][j] = np.sqrt(DeltaX**2 + DeltaY**2 + DeltaZ**2)
    return distance_matrix