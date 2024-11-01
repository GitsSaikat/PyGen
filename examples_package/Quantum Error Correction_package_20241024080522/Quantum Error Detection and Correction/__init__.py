from __future__ import division, print_function
from QEDC import QuantumErrorDetectionCorrection
from sympy import *

class QEDC(object):
    '''
    QEDC is a class with methods to detect and correct quantum errors.
    '''
    def __init__(self, error_rate, num_rep=1):
        self.error_rate = error_rate
        self.num_rep = num_rep
        
    def stabilizer_code(self):
        return QuantumErrorDetectionCorrection(self, 3)
        
    def surface_code(self):
        return QuantumErrorDetectionCorrection(self, 4)
        
    def shor_code(self):
        return QuantumErrorDetectionCorrection(self, 9)

# Use them:
QEC_stab = QEDC(0.1, 3).stabilizer_code()
QEC_sur = QEDC(0.1, 4).surface_code()
QEC_shor = QEDC(0.1, 9).shor_code()
print(QEC_stab.stabilizer_code())
print(QEC_sur.surface_code())
print(QEC_shor.shor_code())