from ase.build import fcc111
from ase.constraints import FixAtoms
import ase.io as aio

slab = fcc111('Pt', size=(3, 3, 3), a=3.924, vacuum=10.0)

mask = [atom.tag < 3 for atom in slab]      # tags 1 & 2 → bottom two layers
slab.set_constraint(FixAtoms(mask=mask))

slab.center(axis=2, vacuum=10.0)
aio.write('Pt111_3x3x3.traj', slab)
print("Wrote slab with", len(slab), "atoms;",
      sum(mask), "are fixed")
