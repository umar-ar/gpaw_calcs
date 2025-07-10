import numpy as np, matplotlib.pyplot as plt
import numpy as np, math
d = np.load('NO_PES.npz')
r, E = d['r'], d['E']

# 0) show raw arrays
print("r-grid (Å):", r)
print("E (eV):    ", E)

# 1) subtract the minimum so numbers are small
i0 = E.argmin()
r0, E0 = r[i0], E[i0]
dE = E - E0
print("\nΔE around minimum:", dE)

# 2) fit window
mask = abs(r - r0) < 0.15
print("\nFit mask:", mask, "→ using", mask.sum(), "points")
print("Points used for fit (r, ΔE):")
for R, dEi in zip(r[mask], dE[mask]):
    print("  %.3f  %.6f" % (R, dEi))

# 3) polynomial fit
if mask.sum() < 3:
    print("**ERROR:** not enough points in mask")
else:
    a2, a1, a0 = np.polyfit(r[mask] - r0, dE[mask], 2)
    k = 2 * a2
    print("\nQuadratic coeff 2*a2 = k =", k, "eV/Å²")
    mu_amu = 1/(1/14.01 + 1/16.00)      # reduced mass of NO
    mu_kg  = mu_amu * 1.660539066e-27   # to kg
    k_SI   = k * 1.602176634e-19 / 1e-20   # N m-1
    omega  = math.sqrt(k_SI / mu_kg)          # rad s-1
    hbar_w = 1.054571817e-34 * omega / 1.602176634e-19  # eV
    print("ħω₀ = %.3f eV  (%.0f meV)" % (hbar_w, 1e3*hbar_w))


# 4) quick plot
plt.plot(r, dE, 'o-')
plt.axvline(r0, ls='--')
plt.xlabel('N–O bond (Å)'); plt.ylabel('E – E₀ (eV)')
plt.tight_layout(); plt.show()

