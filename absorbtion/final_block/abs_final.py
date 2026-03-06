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
    "font.size": 12,
})

def style_axes(ax):
    ax.grid(False)
    ax.tick_params(direction="out", length=6, width=1.2, labelsize=11)
    for sp in ax.spines.values():
        sp.set_linewidth(1.2)

# ============================================================
# BASELINE PARAMETERS
# ============================================================
HP0 = 7.0
BR0 = 5.0
S0  = 3.0
WT0 = 5.0

# ============================================================
# HELPERS
# ============================================================
def gaussian(x, mu, sig):
    return np.exp(-0.5 * ((x - mu) / sig) ** 2)

def logistic(x):
    return 1.0 / (1.0 + np.exp(-x))

# ============================================================
# REDUCED PHYSICS MODEL
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

    # geometry-dependent transition energies
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

    # host absorption
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

    # long-wave drop
    drop = 1.0 - 0.78 * logistic(0.09 * (lam - 1115.0))
    A *= drop

    # oscillatory tail
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

# ============================================================
# SWEEPS
# ============================================================
hp_vals = np.array([2, 4, 6, 8, 10, 12], dtype=float)
height_curves = [absorption_spectrum(lam, hp, BR0, S0, WT0) for hp in hp_vals]
height_labels = [rf"$h_p={hp:.0f}$ nm" for hp in hp_vals]

br_vals = np.array([3, 4, 5, 6, 7, 8], dtype=float)
radius_curves = [absorption_spectrum(lam, HP0, br, S0, WT0) for br in br_vals]
radius_labels = [rf"$b_r={br:.0f}$ nm" for br in br_vals]

br_scale = np.array([3, 4, 5, 6, 7, 8], dtype=float)
hp_scale = (HP0 / BR0) * br_scale
const_aspect_curves = [absorption_spectrum(lam, hp, br, S0, WT0) for hp, br in zip(hp_scale, br_scale)]
const_aspect_labels = [rf"$h_p={hp:.1f},\,b_r={br:.0f}$" for hp, br in zip(hp_scale, br_scale)]

ar_vals = np.array([0.8, 1.1, 1.4, 1.7, 2.0], dtype=float)
hp_ar = ar_vals * BR0
aspect_curves = [absorption_spectrum(lam, hp, BR0, S0, WT0) for hp in hp_ar]
aspect_labels = [rf"$h_p/b_r={ar:.1f}$" for ar in ar_vals]

wt_vals = np.array([2, 3, 4, 5, 6, 7, 8], dtype=float)
wetting_curves = [absorption_spectrum(lam, HP0, BR0, S0, wt) for wt in wt_vals]
wetting_labels = [rf"$w_t={wt:.0f}$ nm" for wt in wt_vals]

# ============================================================
# FIGURE 1: baseline absorption
# ============================================================
fig, ax = plt.subplots(figsize=(7.2, 6.0), dpi=250)
ax.plot(lam, A_base, linewidth=3.0)
ax.plot(lam[::60], A_base[::60], "o", markersize=5.8)
ax.set_xlim(300, 1300)
ax.set_ylim(0.0, 1.0)
ax.set_xlabel("Wavelength (nm)", fontsize=16)
ax.set_ylabel("Absorption", fontsize=18)
ax.set_title("Baseline absorption of QD-IBSC", fontsize=18, pad=12)
style_axes(ax)
plt.tight_layout()
plt.savefig("00_baseline_absorption.png", dpi=600, bbox_inches="tight", facecolor="white")
plt.close()

# ============================================================
# FIGURE 2: raw absorption sweeps
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(15, 9), dpi=250)
axes = axes.flatten()

families = [
    (height_curves, height_labels, "Height sweep"),
    (radius_curves, radius_labels, "Radius sweep"),
    (const_aspect_curves, const_aspect_labels, "Constant aspect-ratio scaling"),
    (aspect_curves, aspect_labels, "Aspect-ratio sweep"),
    (wetting_curves, wetting_labels, "Wetting-layer sweep"),
]

for i, (curves, labels, title) in enumerate(families):
    ax = axes[i]
    for y, lab in zip(curves, labels):
        ax.plot(lam, y, linewidth=2.0, label=lab)
    ax.set_xlim(300, 1300)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Absorption")
    ax.set_title(title)
    ax.legend(fontsize=8, frameon=True)
    style_axes(ax)

# leave last panel for baseline
axes[5].plot(lam, A_base, color="black", linewidth=2.5, label="Baseline")
axes[5].set_xlim(300, 1300)
axes[5].set_ylim(0.0, 1.0)
axes[5].set_xlabel("Wavelength (nm)")
axes[5].set_ylabel("Absorption")
axes[5].set_title("Baseline")
axes[5].legend(fontsize=8, frameon=True)
style_axes(axes[5])

plt.tight_layout()
plt.savefig("01_absorption_raw_sweeps.png", dpi=600, bbox_inches="tight", facecolor="white")
plt.close()

# ============================================================
# FIGURE 3: DIFFERENTIAL ABSORPTION (the useful one)
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(15, 9), dpi=250)
axes = axes.flatten()

for i, (curves, labels, title) in enumerate(families):
    ax = axes[i]
    for y, lab in zip(curves, labels):
        dA = y - A_base
        ax.plot(lam, dA, linewidth=2.0, label=lab)
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
    ax.set_xlim(850, 1300)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel(r"$\Delta A$")
    ax.set_title(title)
    ax.legend(fontsize=8, frameon=True)
    style_axes(ax)

# last panel: zoomed baseline for reference
axes[5].plot(lam, A_base, color="black", linewidth=2.5)
axes[5].set_xlim(850, 1300)
axes[5].set_ylim(0.0, 0.35)
axes[5].set_xlabel("Wavelength (nm)")
axes[5].set_ylabel("Absorption")
axes[5].set_title("Baseline zoom (850–1300 nm)")
style_axes(axes[5])

plt.tight_layout()
plt.savefig("02_absorption_delta_sweeps.png", dpi=600, bbox_inches="tight", facecolor="white")
plt.close()

print("Saved:")
print("  00_baseline_absorption.png")
print("  01_absorption_raw_sweeps.png")
print("  02_absorption_delta_sweeps.png")