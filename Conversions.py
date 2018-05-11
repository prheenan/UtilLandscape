# force floating point division. Can still use integer with //
from __future__ import division
# other good compatibility recquirements for python3
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
# This file is used for importing the common utilities classes.
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.optimize import fminbound,newton,brentq

def kcal_per_mol_per_J():
    """
    :return: the conversion from J/molecule to kcal/mol
    """
    # 1 kcal per mol = 4.184 kJ/mol
    # 1 kcal per mol = (4.184e3 J / molecule) * (1/Avogadro)
    #                ~ 6.947e-21 J / molecule
    # in other words, to convert from J/molecule to kcal/mol, multiply by
    # ~(1 kcal/mol)/(6.947e-21 J / molecule)
    return 1/6.9477e-21