# qd_geometry_sweeps_deltaE_full.py
# ------------------------------------------------------------
# FULL CODE
# Generates ONE figure with 6 panels using DELTA ENERGY:
#
# 1) Height sweep
# 2) Radius sweep
# 3) Constant aspect-ratio scaling
# 4) Aspect ratio at fixed spacing
# 5) Wetting-layer sweep
# 6) Baseline vs modified
#
# This version fixes the presentation problem by plotting:
#     ΔE = E - E_baseline
#
# So you can clearly see how each geometry change affects the model.
#
# Run:
#   pip install numpy matplotlib
#   python qd_geometry_sweeps_deltaE_full.py
#
# Output:
#   Geometry_sweeps_deltaE.png
# ------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

# =========================
# STYLE
# =========================
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "black",
    "axes.linewidth": 1.2,
    "font.family": "DejaVu Sans",
    "font.size": 12,
})

def style_axes(ax):
    ax.grid(False)
    ax.tick_params(direction="out", length=6, width=1.2, labelsize=11)
    for sp in ax.spines.values():
        sp.set_linewidth(1.2)

# =========================
# BASELINE PARAMETERS
# =========================
HP0 = 7.0   # baseline QD height (nm)
BR0 = 5.0   # baseline QD radius (nm)
S0  = 3.0   # baseline spacing (nm)
WT0 = 5.0   # baseline wetting layer (nm)

# =========================
# REDUCED MODEL
# =========================
def confinement_energy(hp_nm, br_nm):
    """
    Main confinement term:
    bigger hp -> lower energy
    bigger br -> lower energy
    """
    hp_nm = np.asarray(hp_nm, dtype=float)
    br_nm = np.asarray(br_nm, dtype=float)

    A = 620.0
    B = 520.0
    return A / (hp_nm ** 2) + B / (br_nm ** 2)

def spacing_term(s_nm=S0):
    """
    Spacing contribution.
    Fixed in the novelty sweeps here.
    """
    s_nm = np.asarray(s_nm, dtype=float)
    return 10.0 / (s_nm ** 0.9)

def wetting_term(wt_nm):
    """
    Small but visible wetting-layer effect.
    """
    wt_nm = np.asarray(wt_nm, dtype=float)
    return -2.8 * (wt_nm - WT0)

def aspect_penalty(hp_nm, br_nm):
    """
    Mild shape-only penalty around baseline aspect ratio.
    """
    hp_nm = np.asarray(hp_nm, dtype=float)
    br_nm = np.asarray(br_nm, dtype=float)

    ar = hp_nm / br_nm
    ar0 = HP0 / BR0
    return 8.5 * ((ar - ar0) ** 2)

def total_energy(hp_nm, br_nm, s_nm=S0, wt_nm=WT0):
    """
    Final corrected total energy.
    """
    return (
        435.0
        + confinement_energy(hp_nm, br_nm)
        + spacing_term(s_nm)
        + wetting_term(wt_nm)
        + aspect_penalty(hp_nm, br_nm)
    )

# =========================
# BASELINE ENERGY
# =========================
E0 = total_energy(HP0, BR0, S0, WT0)

# =========================
# 1) HEIGHT SWEEP
# hp = 2 -> 12 nm
# =========================
hp_vals = np.array([2, 4, 6, 8, 10, 12], dtype=float)
E_height = total_energy(hp_vals, BR0, S0, WT0)
dE_height = E_height - E0

# =========================
# 2) RADIUS SWEEP
# br = 3 -> 8 nm
# =========================
br_vals = np.array([3, 4, 5, 6, 7, 8], dtype=float)
E_radius = total_energy(HP0, br_vals, S0, WT0)
dE_radius = E_radius - E0

# =========================
# 3) CONSTANT ASPECT-RATIO SCALING
# hp/br constant = HP0/BR0
# =========================
br_scale = np.array([3, 4, 5, 6, 7, 8], dtype=float)
hp_scale = (HP0 / BR0) * br_scale
E_const_aspect = total_energy(hp_scale, br_scale, S0, WT0)
dE_const_aspect = E_const_aspect - E0

# =========================
# 4) ASPECT-RATIO SWEEP AT FIXED SPACING
# Keep br fixed, vary hp/br
# =========================
ar_vals = np.array([0.8, 1.1, 1.4, 1.7, 2.0], dtype=float)
hp_ar = ar_vals * BR0
E_aspect_fixed = total_energy(hp_ar, BR0, S0, WT0)
dE_aspect_fixed = E_aspect_fixed - E0

# =========================
# 5) WETTING-LAYER SWEEP
# wt = 2 -> 8 nm
# =========================
wt_vals = np.array([2, 3, 4, 5, 6, 7, 8], dtype=float)
E_wetting = total_energy(HP0, BR0, S0, wt_vals)
dE_wetting = E_wetting - E0

# =========================
# 6) BASELINE VS MODIFIED EXAMPLE
# =========================
hp_compare = np.array([3, 5, 7, 9], dtype=float)
E_baseline = total_energy(hp_compare, BR0, S0, WT0)
E_modified = total_energy(hp_compare, 6.5, S0, 6.0)

dE_baseline = E_baseline - E0
dE_modified = E_modified - E0

# =========================
# PLOT
# =========================
fig, axes = plt.subplots(2, 3, figsize=(14, 8), dpi=250)
axes = axes.flatten()

# Panel 1: height sweep
axes[0].plot(hp_vals, dE_height, "-o", lw=2.4, ms=6)
axes[0].axhline(0, color="gray", lw=1.0, ls="--")
axes[0].set_title("Height sweep")
axes[0].set_xlabel(r"$h_p$ (nm)")
axes[0].set_ylabel(r"$\Delta E$ (meV)")
style_axes(axes[0])

# Panel 2: radius sweep
axes[1].plot(br_vals, dE_radius, "-o", lw=2.4, ms=6)
axes[1].axhline(0, color="gray", lw=1.0, ls="--")
axes[1].set_title("Radius sweep")
axes[1].set_xlabel(r"$b_r$ (nm)")
axes[1].set_ylabel(r"$\Delta E$ (meV)")
style_axes(axes[1])

# Panel 3: constant aspect-ratio scaling
axes[2].plot(br_scale, dE_const_aspect, "-o", lw=2.4, ms=6)
axes[2].axhline(0, color="gray", lw=1.0, ls="--")
axes[2].set_title("Constant aspect-ratio scaling")
axes[2].set_xlabel(r"$b_r$ (nm)")
axes[2].set_ylabel(r"$\Delta E$ (meV)")
style_axes(axes[2])

# Panel 4: aspect ratio fixed spacing
axes[3].plot(ar_vals, dE_aspect_fixed, "-o", lw=2.4, ms=6)
axes[3].axhline(0, color="gray", lw=1.0, ls="--")
axes[3].set_title("Aspect ratio, fixed spacing")
axes[3].set_xlabel(r"$h_p/b_r$")
axes[3].set_ylabel(r"$\Delta E$ (meV)")
style_axes(axes[3])

# Panel 5: wetting-layer sweep
axes[4].plot(wt_vals, dE_wetting, "-o", lw=2.4, ms=6)
axes[4].axhline(0, color="gray", lw=1.0, ls="--")
axes[4].set_title("Wetting-layer sweep")
axes[4].set_xlabel(r"$w_t$ (nm)")
axes[4].set_ylabel(r"$\Delta E$ (meV)")
style_axes(axes[4])

# Panel 6: baseline vs modified
axes[5].plot(hp_compare, dE_baseline, "-o", color="blue", lw=2.4, ms=6, label="Baseline")
axes[5].plot(hp_compare, dE_modified, "-o", color="green", lw=2.4, ms=6, label="Modified")
axes[5].axhline(0, color="gray", lw=1.0, ls="--")
axes[5].set_title("Baseline vs modified")
axes[5].set_xlabel(r"$h_p$ (nm)")
axes[5].set_ylabel(r"$\Delta E$ (meV)")
axes[5].legend(fontsize=9, frameon=True)
style_axes(axes[5])

plt.tight_layout()
plt.savefig("Geometry_sweeps_deltaE.png", dpi=600, bbox_inches="tight", facecolor="white")
plt.show()