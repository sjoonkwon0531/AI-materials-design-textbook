"""Compatibility shim: DScribe 2.1.2 passes ase.Atoms arguments positionally, which
breaks with ase >= 3.26 (a `velocities` argument was inserted). Import this module
before creating MBTR / matrix descriptors of periodic systems. It re-implements
System.__init__ with keyword arguments only; nothing else changes."""
from ase import Atoms
import dscribe.core.system as _dss


def _init(self, symbols=None, positions=None, numbers=None, tags=None, momenta=None,
          masses=None, magmoms=None, charges=None, scaled_positions=None, cell=None,
          pbc=None, celldisp=None, constraint=None, calculator=None, info=None,
          wyckoff_positions=None, equivalent_atoms=None):
    Atoms.__init__(self, symbols=symbols, positions=positions, numbers=numbers, tags=tags,
                   momenta=momenta, masses=masses, magmoms=magmoms, charges=charges,
                   scaled_positions=scaled_positions, cell=cell, pbc=pbc, celldisp=celldisp,
                   constraint=constraint, calculator=calculator, info=info)
    self.wyckoff_positions = wyckoff_positions
    self.equivalent_atoms = equivalent_atoms
    self._cell_inverse = None
    self._displacement_tensor = None
    self._distance_matrix = None
    self._inverse_distance_matrix = None


_dss.System.__init__ = _init
