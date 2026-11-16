# helper shared by ex2_6_5 scripts
import numpy as np
def interface_length_scale(c, Lbox):
    """L_int = area / interface length; interface length from sign changes between neighbours."""
    N = c.shape[0]; dx = Lbox/N; s = c > 0
    P = (np.sum(s != np.roll(s, 1, 0)) + np.sum(s != np.roll(s, 1, 1)))*dx
    return Lbox**2/P
