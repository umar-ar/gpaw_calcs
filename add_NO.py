import ase.io as aio
from ase.build import add_adsorbate, molecule

# Load your slab
slab = aio.read('Pt111_3x3x3.traj')

# Build NO and place it on the hcp hollow site
no = molecule('NO')
add_adsorbate(slab, no, height=1.70, position='hcp')

# Write out the initial adsorbed structure
aio.write('NO_Pt111_init.traj', slab)
print("Wrote NO_Pt111_init.traj with", len(slab), "atoms")

