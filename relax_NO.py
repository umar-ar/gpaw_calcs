import ase.io as aio
from ase.optimize import BFGS
from gpaw import GPAW, PW
from gpaw.poisson import PoissonSolver

# ---- 1. load structure ------------------------------------------------------
atoms = aio.read('NO_Pt111_init.traj')

# ---- 2. GPAW calculator -----------------------------------------------------
calc = GPAW(
    mode=PW(ecut=500),        # 500 eV plane-wave cutoff  (good to <10 meV)
    xc='RPBE',
    kpts=(4, 4, 1),           # 12 irreducible k-points for 3×3 slab
    occupations={'name': 'fermi-dirac', 'width': 0.05},            
    txt='relax.log'           # all SCF output here
)

# spin-polarised by default; GPAW will add a spin channel automatically when O/N present
atoms.calc = calc

# ---- 3. BFGS optimisation ---------------------------------------------------
dyn = BFGS(atoms, trajectory='relax.traj', logfile='relax.BFGS')
dyn.run(fmax=0.03)            # stop when max force < 0.03 eV/Å

# ---- 4. write relaxed geometry ---------------------------------------------
aio.write('NO_Pt111_relaxed.traj', atoms)
# ── now save the converged GPAW state for ΔSCF restarts ────────────────
atoms.calc.write('NO_Pt111_relaxed.gpw', mode='all')
print("Relaxation done → NO_Pt111_relaxed.traj")
