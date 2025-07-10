import numpy as np, ase.io as aio
from ase.constraints import FixBondLength
from ase.optimize  import BFGS
from gpaw          import GPAW, PW

# 1. Load the relaxed NO@Pt structure
ref = aio.read('NO_Pt111_relaxed.traj')
N, O = -2, -1
r0 = ref.get_distance(N, O)

# 2. Define the bond‐length grid
grid = np.arange(r0 - 0.20, r0 + 0.55, 0.05)

# 3. GPAW calculator factory
def make_calc():
    return GPAW(mode=PW(ecut=500),
                xc='RPBE',
                kpts=(4,4,1),
                occupations={'name':'fermi-dirac','width':0.05},
                txt=None)

E = []
for r in grid:
    atoms = ref.copy()

    # 3a) Stretch the N–O bond to exactly r
    posN = atoms[N].position.copy()
    posO = atoms[O].position.copy()
    d = posO - posN
    d /= np.linalg.norm(d)
    atoms.positions[O] = posN + r * d

    # 3b) Now freeze that bond
    bond_constraint = FixBondLength(N, O)
    atoms.set_constraint([bond_constraint, *ref.constraints])

    # 3c) Relax everything else
    atoms.calc = make_calc()
    BFGS(atoms, logfile=None).run(fmax=0.04)

    # 3d) Record energy and structure
    E.append(atoms.get_potential_energy())
    aio.write(f'NO_scan_{r:.2f}.traj', atoms)

# 4. Save the PES
np.savez('NO_PES.npz', r=grid, E=E)
print("Scan done – saved NO_PES.npz with", len(grid), "points")
