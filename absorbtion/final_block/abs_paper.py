import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# STYLE
# ============================================================
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "black",
    "axes.linewidth": 1.2,
    "font.family": "DejaVu Sans",
    "font.size": 11,
})

def style_axes(ax):
    ax.grid(False)
    ax.tick_params(direction="out", length=5, width=1.1, labelsize=10)
    for sp in ax.spines.values():
        sp.set_linewidth(1.1)

def gaussian(x, mu, sig):
    return np.exp(-0.5 * ((x - mu) / sig) ** 2)

def logistic(x):
    return 1.0 / (1.0 + np.exp(-x))

# ============================================================
# BASELINE PARAMETERS
# ============================================================
HP0 = 7.0
BR0 = 5.0
S0  = 3.0
WT0 = 5.0

# ============================================================
# REDUCED MODEL
# ============================================================
def confinement_energy(hp_nm, br_nm):
    hp_nm = np.asarray(hp_nm, dtype=float)
    br_nm = np.asarray(br_nm, dtype=float)
    A = 620.0
    B = 520.0
    return A / (hp_nm ** 2) + B / (br_nm ** 2)

def spacing_term(s_nm=S0):
    s_nm = np.asarray(s_nm, dtype=float)
    return 10.0 / (s_nm ** 0.9)

def wetting_term(wt_nm):
    wt_nm = np.asarray(wt_nm, dtype=float)
    return -2.8 * (wt_nm - WT0)

def aspect_penalty(hp_nm, br_nm):
    hp_nm = np.asarray(hp_nm, dtype=float)
    br_nm = np.asarray(br_nm, dtype=float)
    ar = hp_nm / br_nm
    ar0 = HP0 / BR0
    return 8.5 * ((ar - ar0) ** 2)

def total_energy(hp_nm, br_nm, s_nm=S0, wt_nm=WT0):
    return (
        435.0
        + confinement_energy(hp_nm, br_nm)
        + spacing_term(s_nm)
        + wetting_term(wt_nm)
        + aspect_penalty(hp_nm, br_nm)
    )

def transition_energies_eV(hp_nm, br_nm, s_nm, wt_nm):
    E_meV = total_energy(hp_nm, br_nm, s_nm, wt_nm)
    E_eV = E_meV / 1000.0
    E1 = np.clip(1.02 + 0.55 * (E_eV - 0.40) - 0.015 * (wt_nm - WT0), 0.85, 1.25)
    E2 = np.clip(0.92 + 0.30 * (E_eV - 0.40) - 0.010 * (wt_nm - WT0), 0.75, 1.10)
    return E1, E2

def absorption_amplitude(hp_nm, br_nm, wt_nm):
    size0 = HP0 * (BR0 ** 2)
    size = hp_nm * (br_nm ** 2)
    size_factor = (size / size0) ** 0.18

    ar = hp_nm / br_nm
    ar0 = HP0 / BR0
    ar_factor = 1.0 - 0.10 * max((ar / ar0) - 1.0, 0.0)

    wt_factor = 1.0 + 0.06 * (wt_nm - WT0)
    return 0.82 * size_factor * ar_factor * wt_factor

def absorption_spectrum(lambda_nm, hp_nm, br_nm, s_nm=S0, wt_nm=WT0):
    lam = np.asarray(lambda_nm, dtype=float)
    E = 1240.0 / lam

    host = 0.82 * logistic(10.0 * (E - 1.55))

    E1, E2 = transition_energies_eV(hp_nm, br_nm, s_nm, wt_nm)
    A0 = absorption_amplitude(hp_nm, br_nm, wt_nm)

    band1 = 0.30 * A0 * gaussian(E, E1, 0.08)
    band2 = 0.24 * A0 * gaussian(E, E2, 0.10)
    band3 = 0.18 * A0 * gaussian(E, 1.75, 0.10)

    ripple = 0.05 * A0 * (
        np.sin(2.0 * np.pi * (lam - 620.0) / 170.0)
        + 0.55 * np.sin(2.0 * np.pi * (lam - 620.0) / 85.0)
    )
    ripple *= ((lam >= 500) & (lam <= 1100))

    A = host + band1 + band2 + band3 + ripple

    drop = 1.0 - 0.78 * logistic(0.09 * (lam - 1115.0))
    A *= drop

    tail = lam >= 1120
    A[tail] += (
        0.08 * np.sin(2.0 * np.pi * (lam[tail] - 1120.0) / 42.0)
        + 0.05 * np.sin(2.0 * np.pi * (lam[tail] - 1120.0) / 21.0)
    )

    return np.clip(A, 0.0, 1.0)

# ============================================================
# BASELINE
# ============================================================
lam = np.linspace(300, 1300, 1400)
A_base = absorption_spectrum(lam, HP0, BR0, S0, WT0)
E_base = total_energy(HP0, BR0, S0, WT0)

# ============================================================
# SWEEPS
# ============================================================
hp_vals = np.array([2, 4, 6, 8, 10, 12], dtype=float)
br_vals = np.array([3, 4, 5, 6, 7, 8], dtype=float)
br_scale = np.array([3, 4, 5, 6, 7, 8], dtype=float)
hp_scale = (HP0 / BR0) * br_scale
ar_vals = np.array([0.8, 1.1, 1.4, 1.7, 2.0], dtype=float)
hp_ar = ar_vals * BR0
wt_vals = np.array([2, 3, 4, 5, 6, 7, 8], dtype=float)

# energy sweeps
E_height = total_energy(hp_vals, BR0, S0, WT0) - E_base
E_radius = total_energy(HP0, br_vals, S0, WT0) - E_base
E_const_aspect = total_energy(hp_scale, br_scale, S0, WT0) - E_base
E_aspect = total_energy(hp_ar, BR0, S0, WT0) - E_base
E_wetting = total_energy(HP0, BR0, S0, wt_vals) - E_base

# absorption sweeps
A_height = [absorption_spectrum(lam, hp, BR0, S0, WT0) - A_base for hp in hp_vals]
A_radius = [absorption_spectrum(lam, HP0, br, S0, WT0) - A_base for br in br_vals]
A_const_aspect = [absorption_spectrum(lam, hp, br, S0, WT0) - A_base for hp, br in zip(hp_scale, br_scale)]
A_aspect = [absorption_spectrum(lam, hp, BR0, S0, WT0) - A_base for hp in hp_ar]
A_wetting = [absorption_spectrum(lam, HP0, BR0, S0, wt) - A_base for wt in wt_vals]

# ============================================================
# FINAL COMBINED FIGURE
# ============================================================
fig, axes = plt.subplots(5, 2, figsize=(13, 18), dpi=250)

rows = [
    ("(a) Height sweep", hp_vals, E_height, A_height, [rf"$h_p={v:.0f}$" for v in hp_vals], r"$h_p$ (nm)"),
    ("(b) Radius sweep", br_vals, E_radius, A_radius, [rf"$b_r={v:.0f}$" for v in br_vals], r"$b_r$ (nm)"),
    ("(c) Constant aspect-ratio scaling", br_scale, E_const_aspect, A_const_aspect,
     [rf"$h_p={hp:.1f},\,b_r={br:.0f}$" for hp, br in zip(hp_scale, br_scale)], r"$b_r$ (nm)"),
    ("(d) Aspect-ratio sweep", ar_vals, E_aspect, A_aspect, [rf"$h_p/b_r={v:.1f}$" for v in ar_vals], r"$h_p/b_r$"),
    ("(e) Wetting-layer sweep", wt_vals, E_wetting, A_wetting, [rf"$w_t={v:.0f}$" for v in wt_vals], r"$w_t$ (nm)")
]

for i, (title, xvals, dE, dA_family, labels, xlabel) in enumerate(rows):
    # left: energy shift
    axL = axes[i, 0]
    axL.plot(xvals, dE, "-o", lw=2.4, ms=5.5)
    axL.axhline(0.0, color="gray", lw=1.0, ls="--")
    axL.set_title(title, fontsize=14)
    axL.set_xlabel(xlabel, fontsize=12)
    axL.set_ylabel(r"Energy shift, $\Delta E$ (meV)", fontsize=12)
    style_axes(axL)

    # right: differential absorption
    axR = axes[i, 1]
    for y, lab in zip(dA_family, labels):
        axR.plot(lam, y, lw=1.9, label=lab)
    axR.axhline(0.0, color="gray", lw=1.0, ls="--")
    axR.set_xlim(850, 1300)
    axR.set_xlabel("Wavelength (nm)", fontsize=12)
    axR.set_ylabel(r"Differential absorption, $\Delta A$", fontsize=12)
    axR.legend(fontsize=7.5, frameon=True, ncol=1)
    style_axes(axR)

plt.tight_layout()
plt.savefig("Final_combined_results_QD_IBSC.png", dpi=600, bbox_inches="tight", facecolor="white")
plt.show()