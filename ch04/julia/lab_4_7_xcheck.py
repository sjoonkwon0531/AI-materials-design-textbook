# Mirror of lab_4_7.jl's layer-cut logic (z from get_positions, rounded, sorted unique, 2nd entry) for all sizes/vacua used
import numpy as np
from ase.build import fcc111
for size,vac in [((2,2,4),8.0),((2,2,3),8.0),((2,2,6),8.0),((2,2,4),5.0),((2,2,4),12.0)]:
    s=fcc111('Pt',size=size,vacuum=vac); z=s.get_positions()[:,2]
    lay=np.sort(np.unique(np.round(z,2))); cut=lay[1]+0.1
    print(size,vac,"fixed atoms:",int((z<cut).sum()),"of",len(s))
