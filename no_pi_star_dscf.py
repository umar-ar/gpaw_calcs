#!/usr/bin/env python3
"""
no_pi_star_dscf_gpaw26_serialSearch.py  (root‑only π* search, GPAW 25.1 safe)
-----------------------------------------------------------------------------
Runs the same physics workflow as the original ∆SCF script but avoids every
MPI‑locality pitfall **and** the missing `collect_wave_functions()` method in
GPAW 25.1 by relying on `calc.initialize()` to make all k‑point data available
on rank 0 during the π*-band hunt.

Execute with parallel SCF but serial π* detection:
    mpirun -np 6 gpaw python no_pi_star_dscf_gpaw26_serialSearch.py
"""
from __future__ import annotations

import resource
from pathlib import Path
from typing import List, Tuple

import numpy as np
from ase.build import fcc111, molecule
from ase.constraints import FixAtoms
from ase.parallel import parprint, world
from gpaw import GPAW, FermiDirac, PoissonSolver, restart
import gpaw.dscf as dscf

# ---------------------------------------------------------------------------
# Global switches
# ---------------------------------------------------------------------------
VERBOSE = True            # master switch for detailed logging
WEIGHT_CUTOFF = 0.20      # Σ|P|² threshold to identify the π* candidate

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def vprint(*args, **kwargs):
    if VERBOSE and world.rank == 0:
        parprint(*args, **kwargs)

def mem_usage_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if rss > 1 << 30:  # macOS returns bytes
        rss /= 1024
    return rss / 1024  # MiB

# ---------------------------------------------------------------------------
# Projector helpers
# ---------------------------------------------------------------------------

def pxy_projector_rows(calc: GPAW, aidx: int) -> List[int]:
    """Indices of p‑projector rows for atom *aidx*."""
    setup = calc.wfs.setups[aidx]
    return [j for j, l in enumerate(setup.l_j) if l == 1]


def find_pi_star_band_serial(calc: GPAW, Nidx: int, Oidx: int) -> np.ndarray:
    """Locate the NO 2π* band at each k‑point (root rank only)."""
    nk = len(calc.wfs.kpt_u)
    band_k = np.empty(nk, dtype=np.int32)

    if world.rank == 0:
        calc.initialize()  # ensure full wave‑function data on root
        fermi = calc.get_fermi_level()
        pN, pO = pxy_projector_rows(calc, Nidx), pxy_projector_rows(calc, Oidx)

        vprint("Searching for π* candidates … (root)")
        for ik, kpt in enumerate(calc.wfs.kpt_u):
            f_n = kpt.f_n        # full occupations
            e_n = kpt.e_n        # eigenvalues (eV)
            P_ani = kpt.P_ani    # projector coefficients

            best_n, best_w = -1, 0.0
            for n, occ in enumerate(f_n):
                if occ > 1e-3:  # occupied → skip
                    continue
                w = 0.0
                for a, rows in ((Nidx, pN), (Oidx, pO)):
                    P_ai = P_ani.get(a)
                    if P_ai is not None:
                        w += np.sum(np.abs(P_ai[rows, n])**2)
                if w > best_w:
                    best_n, best_w = n, w
                if w >= WEIGHT_CUTOFF:
                    break
            if best_n == -1:
                raise RuntimeError(f"No π* band at k‑point {ik}, max w={best_w:.2f}")
            band_k[ik] = best_n
            vprint(f"  k={ik:2d}: band={best_n:3d}  w={best_w:.2f}  Δε={e_n[best_n]-fermi:+6.3f} eV")

    world.broadcast(band_k, 0)
    return band_k


def gather_pi_star_data(calc: GPAW, band_k: np.ndarray,
                         Nidx: int, Oidx: int) -> Tuple[list, list]:
    """Gather π* wave‑function slices + projectors across MPI ranks."""
    wf_u, p_uai = [], []
    bd = calc.wfs.bd

    for ik, kpt in enumerate(calc.wfs.kpt_u):
        n_gl = int(band_k[ik])
        owner, n_loc = bd.who_has(n_gl)
        if owner == bd.comm.rank:
            wf_u.append(kpt.psit_nG[n_loc])
            proj = {}
            for a in (Nidx, Oidx):
                P_ai = kpt.P_ani.get(a)
                if P_ai is not None:
                    proj[a] = P_ai[:, n_loc]
            p_uai.append(proj)
        else:
            wf_u.append(None)
            p_uai.append({})

    if world.rank == 0:
        norm = sum((abs(p)**2).sum() for d in p_uai for p in d.values())
        vprint(f"[diag] Σ|P|²(N/O) = {norm:.3f}")
    return wf_u, p_uai

# ---------------------------------------------------------------------------
# Build Pt(111) slab + NO adsorbate
# ---------------------------------------------------------------------------
slab = fcc111('Pt', size=(2, 2, 4), a=3.924, vacuum=12.0, orthogonal=True)
slab.center(axis=2, vacuum=12.0)
no = molecule('NO'); no.rotate(90, 'y'); no.translate(slab.positions[-1] + (0, 0, 1.8))
atoms = slab + no; atoms.pbc = (True, True, False)
mask = [i < 8 for i in range(len(slab))] + [False, False]
atoms.set_constraint(FixAtoms(mask=mask))
Nidx, Oidx = len(atoms) - 2, len(atoms) - 1

# ---------------------------------------------------------------------------
# Ground‑state calculation
# ---------------------------------------------------------------------------
calc_gs = GPAW(h=0.18, xc='PBE', kpts=(4,4,1), spinpol=True,
               occupations=FermiDirac(0.10),
               poissonsolver=PoissonSolver(name='fast', dipolelayer=2),
               symmetry={'point_group': False},
               convergence={'energy': 1e-5}, txt='gs.log')

atoms.calc = calc_gs
E_gs = atoms.get_potential_energy(); calc_gs.write('no_pt_gs.gpw', mode='all')
vprint(f"E_gs = {E_gs:.6f} eV  (RSS {mem_usage_mb():.0f} MiB)")

# ---------------------------------------------------------------------------
# Restart & π* detection
# ---------------------------------------------------------------------------
_atoms, calc = restart('no_pt_gs.gpw', txt='-')
assert calc.wfs.mode == 'fd'

band_k = find_pi_star_band_serial(calc, Nidx, Oidx)
vprint('band_k:', band_k.tolist())

wf_u, p_uai = gather_pi_star_data(calc, band_k, Nidx, Oidx)

# ---------------------------------------------------------------------------
# ∆SCF excited‑state
# ---------------------------------------------------------------------------
calc_es = calc.new(txt='es.log'); _atoms.calc = calc_es
orbital = dscf.AEOrbital(calc_es, wf_u, p_uai)

vprint('Starting ∆SCF …  (RSS {:.0f} MiB)'.format(mem_usage_mb()))
dscf.dscf_calculation(calc_es, [[1.0, orbital, 1]], _atoms)
E_es = _atoms.get_potential_energy()

parprint(f"\nVertical ∆E(NO 2π*) = {E_es - E_gs:.3f} eV")

if world.rank == 0:
    Path('summary.txt').write_text(
        f"E_gs = {E_gs:.6f} eV\nE_es = {E_es:.6f} eV\n∆E  = {E_es - E_gs:.6f} eV\n")
    vprint('Summary written to summary.txt')
