# fig3_perfect_m0_m1_truncated_cone_FIXED.py
# Q2-level Fig.3 reproduction (Python/VS Code) with boundary-artifact suppression.
#
# Output panels:
# (a) m=0 ground-state wavefunction  ψ0(x,y) at z-slice
# (b) m=1 first-excited (p-like) wavefunction ψ1(x,y) at same slice (two lobes)
# (c) |ψ0|^2
# (d) |ψ1|^2
#
# Run:
#   pip install numpy scipy matplotlib
#   python fig3_perfect_m0_m1_truncated_cone_FIXED.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

# =========================
# 1) EDIT ONLY THESE (YOUR SWEEPS)
# =========================
# Geometry (nm)
hp_nm = 10.0        # QD height (sweep A)
br_nm = 6.0         # bottom radius (sweep B)
tr_nm = 3.5         # top radius (aspect/truncation control)
wt_nm = 5.0         # wetting layer thickness (sweep D)

# Surrounding GaAs thickness (increase to kill boundary rings)
sub_nm = 25.0       # GaAs below wetting
cap_nm = 25.0       # GaAs cap above dot

# Domain (nm) - large enough to avoid reflections
Rmax_nm = 80.0
Zmax_nm = sub_nm + wt_nm + hp_nm + cap_nm

# Grid (higher = smoother; keep moderate if your PC is weak)
Nr = 340
Nz = 380

# Potential (eV): GaAs barrier = 0, QD+WL = -V0
V0_eV = 0.55

# Absorbing wall near boundaries to remove concentric ring artifacts
use_absorbing_wall = True
Vwall_eV = 6.0      # stronger wall
wall_frac = 0.25    # thicker wall region

# Slice to plot (nm): "mid_dot" or numeric
slice_location = "mid_dot"

# p-like orientation: "vertical" -> top/bottom lobes (paper-like)
p_lobe_orientation = "vertical"

# Plot window (nm) for x–y maps (keep ~30 nm like paper)
Lxy_nm = 30.0
Nxy = 520           # higher -> smoother circles

# Output
out_png = "Fig3_Q2_truncated_cone_m0_m1_FIXED.png"
dpi = 350

# =========================
# 2) GRID
# =========================
r = np.linspace(0.0, Rmax_nm, Nr)
z = np.linspace(0.0, Zmax_nm, Nz)
dr = r[1] - r[0]
dz = z[1] - z[0]
R, Z = np.meshgrid(r, z, indexing="xy")  # (Nz, Nr)

# Placement
z_sub_top = sub_nm
z_wet_top = sub_nm + wt_nm
z_dot_top = z_wet_top + hp_nm

def cone_radius(zval_nm: np.ndarray) -> np.ndarray:
    t = (zval_nm - z_wet_top) / max(hp_nm, 1e-12)
    return br_nm + (tr_nm - br_nm) * t

# Masks (truncated cone + wetting layer)
wet_mask = (Z >= z_sub_top) & (Z <= z_wet_top) & (R <= br_nm)
dot_mask = (Z >= z_wet_top) & (Z <= z_dot_top) & (R <= cone_radius(Z))
qd_mask = wet_mask | dot_mask

V = np.zeros((Nz, Nr), dtype=float)
V[qd_mask] = -V0_eV

# Smooth absorbing wall to suppress boundary reflections
if use_absorbing_wall:
    r0 = Rmax_nm * (1.0 - wall_frac)
    z0_bot = Zmax_nm * wall_frac
    z0_top = Zmax_nm * (1.0 - wall_frac)

    wall = np.zeros_like(V)

    # radial wall
    wr = np.clip((R - r0) / max(Rmax_nm - r0, 1e-12), 0.0, 1.0)
    wall += (wr**4)

    # bottom wall
    wz_bot = np.clip((z0_bot - Z) / max(z0_bot, 1e-12), 0.0, 1.0)
    wall += (wz_bot**4)

    # top wall
    wz_top = np.clip((Z - z0_top) / max(Zmax_nm - z0_top, 1e-12), 0.0, 1.0)
    wall += (wz_top**4)

    V += Vwall_eV * wall

# =========================
# 3) SPARSE HAMILTONIAN FOR GIVEN m (AXISYMMETRIC)
#     Hψ = [-(∂²/∂r² + (1/r)∂/∂r + ∂²/∂z²) + (m²/r²)]ψ + Vψ
# BCs:
#   z=0, z=Zmax, r=Rmax -> Dirichlet
#   r=0: m=0 Neumann; m>0 Dirichlet
# =========================
def build_H_for_m(m: int) -> csr_matrix:
    inv_dr2 = 1.0 / (dr * dr)
    inv_dz2 = 1.0 / (dz * dz)
    N = Nr * Nz

    rows, cols, data = [], [], []

    def add(i, j, v):
        rows.append(i); cols.append(j); data.append(v)

    for iz in range(Nz):
        for ir_i in range(Nr):
            idx = iz * Nr + ir_i

            # Dirichlet boundaries
            if ir_i == Nr - 1 or iz == 0 or iz == Nz - 1:
                add(idx, idx, 1.0)
                continue

            # r=0
            if ir_i == 0:
                if m == 0:
                    idx_rp = iz * Nr + 1
                    diag = V[iz, 0] + 2.0 * inv_dz2 + 2.0 * inv_dr2
                    add(idx, (iz - 1) * Nr + 0, -inv_dz2)
                    add(idx, (iz + 1) * Nr + 0, -inv_dz2)
                    add(idx, idx_rp, -2.0 * inv_dr2)
                    add(idx, idx, diag)
                else:
                    add(idx, idx, 1.0)
                continue

            ri = r[ir_i]
            idx_rm = iz * Nr + (ir_i - 1)
            idx_rp = iz * Nr + (ir_i + 1)
            idx_zm = (iz - 1) * Nr + ir_i
            idx_zp = (iz + 1) * Nr + ir_i

            diag = V[iz, ir_i]

            # z second derivative
            diag += 2.0 * inv_dz2
            add(idx, idx_zm, -inv_dz2)
            add(idx, idx_zp, -inv_dz2)

            # radial d2/dr2
            diag += 2.0 * inv_dr2
            add(idx, idx_rm, -inv_dr2)
            add(idx, idx_rp, -inv_dr2)

            # (1/r) d/dr
            add(idx, idx_rp, -(1.0 / (2.0 * ri)) * (1.0 / dr))
            add(idx, idx_rm, +(1.0 / (2.0 * ri)) * (1.0 / dr))

            # m^2 / r^2
            if m != 0:
                diag += (m * m) / (ri * ri)

            add(idx, idx, diag)

    return csr_matrix((data, (rows, cols)), shape=(N, N))

def axisym_normalize(psi_rz: np.ndarray) -> np.ndarray:
    w = r[np.newaxis, :] * dr * dz
    nrm = np.sqrt(np.sum(np.abs(psi_rz)**2 * w))
    return psi_rz / (nrm + 1e-30)

def solve_lowest_state(m: int) -> np.ndarray:
    H = build_H_for_m(m)
    evals, evecs = eigsh(H, k=8, which="SA")
    order = np.argsort(evals)
    psi = evecs[:, order[0]].reshape(Nz, Nr)
    return axisym_normalize(psi)

# =========================
# 4) SOLVE: m=0 and m=1
# =========================
psi_m0_rz = solve_lowest_state(m=0)
psi_m1_rz = solve_lowest_state(m=1)

# =========================
# 5) PICK Z SLICE
# =========================
if slice_location == "mid_dot":
    z_slice_nm = z_wet_top + 0.5 * hp_nm
else:
    z_slice_nm = float(slice_location)

iz_slice = int(np.argmin(np.abs(z - z_slice_nm)))

psi0_r = psi_m0_rz[iz_slice, :]
psi1_r = psi_m1_rz[iz_slice, :]

# QD radius at this slice for overlay
if z_sub_top <= z_slice_nm <= z_wet_top:
    r_qd = br_nm
elif z_wet_top < z_slice_nm <= z_dot_top:
    r_qd = float(cone_radius(np.array(z_slice_nm)))
else:
    r_qd = 0.0

# =========================
# 6) MAP TO X–Y WITH ANGULAR DEPENDENCE
# =========================
x = np.linspace(-Lxy_nm, Lxy_nm, Nxy)
y = np.linspace(-Lxy_nm, Lxy_nm, Nxy)
X2, Y2 = np.meshgrid(x, y, indexing="xy")
R2 = np.sqrt(X2**2 + Y2**2)

psi0_xy = np.interp(R2.ravel(), r, psi0_r, left=0.0, right=0.0).reshape(Nxy, Nxy)
psi1_rad_xy = np.interp(R2.ravel(), r, psi1_r, left=0.0, right=0.0).reshape(Nxy, Nxy)

eps = 1e-30
if p_lobe_orientation.lower() == "vertical":
    ang = Y2 / (R2 + eps)   # sin(theta): top/bottom lobes
else:
    ang = X2 / (R2 + eps)   # cos(theta): left/right lobes

psi1_xy = psi1_rad_xy * ang

p0_xy = np.abs(psi0_xy)**2
p1_xy = np.abs(psi1_xy)**2

mask = R2 > Lxy_nm
psi0_xy = np.ma.array(psi0_xy, mask=mask)
psi1_xy = np.ma.array(psi1_xy, mask=mask)
p0_xy = np.ma.array(p0_xy, mask=mask)
p1_xy = np.ma.array(p1_xy, mask=mask)

# =========================
# 7) PLOT
# =========================
fig, axes = plt.subplots(1, 4, figsize=(15.2, 3.8), constrained_layout=True)

def draw_circle(ax, radius):
    if radius <= 0:
        return
    th = np.linspace(0, 2*np.pi, 900)
    ax.plot(radius*np.cos(th), radius*np.sin(th), lw=1.1, alpha=0.95)

def panel(ax, img, title, symmetric=False):
    if symmetric:
        mmax = float(np.nanmax(np.abs(img)))
        vmin, vmax = -mmax, mmax
    else:
        vmin, vmax = None, None

    im = ax.imshow(
        img, extent=[-Lxy_nm, Lxy_nm, -Lxy_nm, Lxy_nm],
        origin="lower", cmap="jet", vmin=vmin, vmax=vmax
    )
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")
    ax.set_aspect("equal")
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=9)
    draw_circle(ax, r_qd)

panel(axes[0], psi0_xy, r"(a) ground state  $\psi_0$ (m=0)", symmetric=True)
panel(axes[1], psi1_xy, r"(b) first excited  $\psi_1$ (m=1)", symmetric=True)
panel(axes[2], p0_xy,   r"(c) ground  $|\psi_0|^2$")
panel(axes[3], p1_xy,   r"(d) excited  $|\psi_1|^2$")

fig.suptitle(
    f"Truncated-cone QD wavefunctions (axisymmetric m=0/m=1) | "
    f"slice z={z_slice_nm:.2f} nm | hp={hp_nm} nm, br={br_nm} nm, tr={tr_nm} nm, wt={wt_nm} nm",
    fontsize=11
)

plt.savefig(out_png, dpi=dpi, bbox_inches="tight")
plt.show()
print("Saved:", out_png)