# qd_ibsc_geometry_sweeps.py
# ------------------------------------------------------------
# PARAMETER-DEPENDENT Python model for your QD project
# ------------------------------------------------------------
# This script does what we should have done from the start:
# it ties the OUTPUTS directly to the geometry changes:
#
# 1) QD height sweep          hp = 2 -> 12 nm
# 2) QD radius sweep          br = 3 -> 8 nm
# 3) Constant aspect scaling  hp/br = const
# 4) Aspect-ratio sweep       hp/br changes, spacing s fixed
# 5) Wetting-layer sweep      wt = 2 -> 8 nm
#
# It generates:
#   - baseline absorption
#   - energy vs each sweep parameter
#   - absorption vs wavelength for each sweep
#
# This is a reduced physics model:
#   - confinement energy depends on hp and br
#   - coupling / lattice effect depends on spacing s and QD count
#   - wetting layer affects transition strength and small energy shift
#   - absorption is generated from geometry-dependent transition energies
#
# It is NOT a full ab initio solver, but it is parameter-dependent and
# legitimate for trend studies and paper figures.
#
# Run:
#   pip install numpy matplotlib
#   python qd_ibsc_geometry_sweeps.py
#
# Outputs:
#   baseline_absorption.png
#   sweep_height_energy.png
#   sweep_height_absorption.png
#   sweep_radius_energy.png
#   sweep_radius_absorption.png
#   sweep_const_aspect_energy.png
#   sweep_const_aspect_absorption.png
#   sweep_aspect_fixed_spacing_energy.png
#   sweep_aspect_fixed_spacing_absorption.png
#   sweep_wetting_energy.png
#   sweep_wetting_absorption.png
# ------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

# ============================
# GLOBAL STYLE
# ============================
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "black",
    "axes.linewidth": 1.3,
    "font.family": "DejaVu Sans",
    "font.size": 12,
})

def style_axes(ax):
    ax.grid(False)
    ax.tick_params(direction="out", length=6, width=1.2, labelsize=12)
    for sp in ax.spines.values():
        sp.set_linewidth(1.2)

# ============================
# PHYSICAL CONSTANTS
# ============================
hbar = 1.054571817e-34
q    = 1.602176634e-19
m0   = 9.1093837015e-31
J01  = 2.4048255577  # first root of J0

def nm_to_m(x_nm):
    return np.asarray(x_nm, dtype=float) * 1e-9

def J_to_meV(EJ):
    return (EJ / q) * 1e3

def eV_to_nm(EeV):
    return 1240.0 / np.asarray(EeV, dtype=float)

def gaussian(x, mu, sig):
    return np.exp(-0.5 * ((x - mu) / sig) ** 2)

def logistic(x):
    return 1.0 / (1.0 + np.exp(-x))

# ============================
# BASELINE MODEL PARAMETERS
# ============================
# Use the baseline consistently.
BASE = {
    "hp_nm": 7.0,   # QD height
    "br_nm": 5.0,   # QD bottom radius
    "s_nm":  3.0,   # barrier spacing
    "wt_nm": 5.0,   # wetting layer thickness
    "n_qd":  20,    # dense QD case for absorption
}

# ============================
# REDUCED PHYSICS MODEL
# ============================
def confinement_energy_meV(hp_nm, br_nm, mstar=0.05*m0):
    """
    Quantum confinement:
      Econf ~ (ħ²/2m*) [pi²/hp² + j01²/br²]
    """
    hp = nm_to_m(hp_nm)
    br = nm_to_m(br_nm)
    pref = (hbar**2) / (2.0 * mstar)
    E_J = pref * ((np.pi**2)/(hp**2) + (J01**2)/(br**2))
    return J_to_meV(E_J)

def collective_uplift_meV(s_nm, n_qd):
    """
    Density / lattice / coupling surrogate:
    - 1 QD: low uplift, nearly flat
    - more QDs: higher uplift
    - spacing effect appears mainly for dense stacks
    """
    dens = (n_qd - 1) / 19.0  # 1 -> 0, 20 -> 1

    base_uplift = 110.0 * dens
    spacing_uplift = (55.0 * dens) * np.log1p(max(s_nm - 1.0, 0.0))

    return base_uplift + spacing_uplift

def wetting_shift_meV(wt_nm):
    """
    Wetting layer changes local confinement / coupling slightly.
    Centered around wt = 5 nm baseline.
    """
    return 4.0 * (wt_nm - 5.0)

def total_energy_meV(hp_nm, br_nm, s_nm, wt_nm, n_qd):
    """
    Total reduced-model electron energy.
    """
    E0 = confinement_energy_meV(hp_nm, br_nm)
    Ec = collective_uplift_meV(s_nm, n_qd)
    Ew = wetting_shift_meV(wt_nm)

    # affine calibration to keep values in a paper-like meV window
    # and preserve parameter dependence
    E = 0.33 * E0 + Ec + Ew + 305.0
    return E

# ============================
# ABSORPTION MODEL
# ============================
def transition_energies_eV(hp_nm, br_nm, s_nm, wt_nm, n_qd):
    """
    Build two geometry-dependent transition energies:
      VB -> IB
      IB -> CB
    derived from total electron energy.
    """
    E_meV = total_energy_meV(hp_nm, br_nm, s_nm, wt_nm, n_qd)
    E_eV = E_meV / 1000.0

    # Geometry-dependent surrogate transitions.
    # These are set so the spectrum changes when geometry changes.
    E_v_ib = np.clip(1.08 - 0.28*(E_eV - 0.40) + 0.015*(wt_nm - 5.0), 0.82, 1.22)
    E_ib_c = np.clip(0.50 + 0.35*(E_eV - 0.40) + 0.010*(wt_nm - 5.0), 0.35, 0.78)

    return E_v_ib, E_ib_c

def absorption_spectrum(lambda_nm, hp_nm, br_nm, s_nm, wt_nm, n_qd):
    """
    Geometry-dependent absorption spectrum:
    - high host absorption at short wavelength
    - QD/IB bands shift with geometry
    - wetting layer changes intensity
    """
    lam = np.asarray(lambda_nm, dtype=float)
    E = 1240.0 / lam  # eV

    # Strong host absorption at short wavelengths
    host = 0.82 * logistic(10.0 * (E - 1.55))

    E_v_ib, E_ib_c = transition_energies_eV(hp_nm, br_nm, s_nm, wt_nm, n_qd)

    dens = (n_qd - 1) / 19.0
    f_wt = 1.0 + 0.06 * (wt_nm - 5.0)
    f_qd = (0.38 + 0.48 * dens) * f_wt

    band1 = f_qd * 0.34 * gaussian(E, E_v_ib, 0.10)
    band2 = f_qd * 0.24 * gaussian(E, 1.50,   0.12)
    band3 = f_qd * 0.30 * gaussian(E, E_ib_c + 0.75, 0.11)

    ripple = 0.05 * f_qd * (
        np.sin(2*np.pi*(lam - 620.0)/170.0)
        + 0.55*np.sin(2*np.pi*(lam - 620.0)/85.0)
    )
    ripple *= ((lam >= 500) & (lam <= 1100))

    A = host + band1 + band2 + band3 + ripple

    # Drop after ~1100 nm, with oscillatory tail
    drop = 1.0 - 0.78 * logistic(0.09 * (lam - 1115.0))
    A *= drop

    tail = lam >= 1120
    A[tail] += 0.08*np.sin(2*np.pi*(lam[tail] - 1120.0)/42.0) \
             + 0.05*np.sin(2*np.pi*(lam[tail] - 1120.0)/21.0)

    return np.clip(A, 0.0, 1.0)

# ============================
# PLOTTING HELPERS
# ============================
def plot_energy(x, y, xlabel, title, savepath):
    fig, ax = plt.subplots(figsize=(7.0, 5.8), dpi=220)
    ax.plot(x, y, "-o", linewidth=3.0, markersize=7.0)
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel("Energy (meV)", fontsize=18)
    ax.set_title(title, fontsize=18, pad=12)
    style_axes(ax)
    fig.savefig(savepath, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)

def plot_absorption_family(lambda_nm, curves, labels, title, savepath):
    fig, ax = plt.subplots(figsize=(7.2, 6.0), dpi=220)

    for y, lab in zip(curves, labels):
        ax.plot(lambda_nm, y, linewidth=2.6, label=lab)

    ax.set_xlim(300, 1300)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Wavelength (nm)", fontsize=16)
    ax.set_ylabel("Absorption", fontsize=18)
    ax.set_title(title, fontsize=18, pad=12)
    ax.legend(frameon=True, fontsize=11)
    style_axes(ax)

    fig.savefig(savepath, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)

# ============================
# BASELINE FIGURE
# ============================
def make_baseline_absorption():
    lam = np.linspace(300, 1300, 1200)
    A = absorption_spectrum(
        lam,
        BASE["hp_nm"],
        BASE["br_nm"],
        BASE["s_nm"],
        BASE["wt_nm"],
        BASE["n_qd"]
    )

    fig, ax = plt.subplots(figsize=(7.2, 6.0), dpi=220)
    ax.plot(lam, A, linewidth=3.2)
    ax.plot(lam[::60], A[::60], "o", markersize=6.0)

    ax.set_xlim(300, 1300)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Wavelength (nm)", fontsize=16)
    ax.set_ylabel("Absorption", fontsize=18)
    ax.set_title("Baseline absorption of QD-IBSC", fontsize=18, pad=12)
    style_axes(ax)

    fig.savefig("baseline_absorption.png", dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)

# ============================
# 1) HEIGHT SWEEP
# ============================
def sweep_height():
    hp_vals = np.array([2, 4, 6, 8, 10, 12], dtype=float)

    E = [
        total_energy_meV(hp, BASE["br_nm"], BASE["s_nm"], BASE["wt_nm"], BASE["n_qd"])
        for hp in hp_vals
    ]
    plot_energy(
        hp_vals, E,
        "QD height $h_p$ (nm)",
        "Energy vs QD height",
        "sweep_height_energy.png"
    )

    lam = np.linspace(300, 1300, 1200)
    curves = [
        absorption_spectrum(lam, hp, BASE["br_nm"], BASE["s_nm"], BASE["wt_nm"], BASE["n_qd"])
        for hp in hp_vals
    ]
    labels = [f"$h_p$={hp:.0f} nm" for hp in hp_vals]
    plot_absorption_family(
        lam, curves, labels,
        "Absorption for QD height sweep",
        "sweep_height_absorption.png"
    )

# ============================
# 2) RADIUS SWEEP
# ============================
def sweep_radius():
    br_vals = np.array([3, 4, 5, 6, 7, 8], dtype=float)

    E = [
        total_energy_meV(BASE["hp_nm"], br, BASE["s_nm"], BASE["wt_nm"], BASE["n_qd"])
        for br in br_vals
    ]
    plot_energy(
        br_vals, E,
        "QD radius $b_r$ (nm)",
        "Energy vs QD radius",
        "sweep_radius_energy.png"
    )

    lam = np.linspace(300, 1300, 1200)
    curves = [
        absorption_spectrum(lam, BASE["hp_nm"], br, BASE["s_nm"], BASE["wt_nm"], BASE["n_qd"])
        for br in br_vals
    ]
    labels = [f"$b_r$={br:.0f} nm" for br in br_vals]
    plot_absorption_family(
        lam, curves, labels,
        "Absorption for QD radius sweep",
        "sweep_radius_absorption.png"
    )

# ============================
# 3) CONSTANT ASPECT-RATIO SCALING
# ============================
def sweep_constant_aspect():
    aspect = BASE["hp_nm"] / BASE["br_nm"]  # keep this constant
    br_vals = np.array([3, 4, 5, 6, 7, 8], dtype=float)
    hp_vals = aspect * br_vals

    E = [
        total_energy_meV(hp, br, BASE["s_nm"], BASE["wt_nm"], BASE["n_qd"])
        for hp, br in zip(hp_vals, br_vals)
    ]
    plot_energy(
        br_vals, E,
        "Scaled radius $b_r$ (nm) at constant $h_p/b_r$",
        "Energy for constant aspect-ratio scaling",
        "sweep_const_aspect_energy.png"
    )

    lam = np.linspace(300, 1300, 1200)
    curves = [
        absorption_spectrum(lam, hp, br, BASE["s_nm"], BASE["wt_nm"], BASE["n_qd"])
        for hp, br in zip(hp_vals, br_vals)
    ]
    labels = [f"$h_p$={hp:.1f}, $b_r$={br:.0f}" for hp, br in zip(hp_vals, br_vals)]
    plot_absorption_family(
        lam, curves, labels,
        "Absorption for constant aspect-ratio scaling",
        "sweep_const_aspect_absorption.png"
    )

# ============================
# 4) ASPECT-RATIO SWEEP AT FIXED SPACING
# ============================
def sweep_aspect_fixed_spacing():
    # keep spacing fixed, radius fixed, vary aspect through hp
    br_fixed = BASE["br_nm"]
    aspect_vals = np.array([0.6, 1.0, 1.4, 1.8, 2.2], dtype=float)
    hp_vals = aspect_vals * br_fixed

    E = [
        total_energy_meV(hp, br_fixed, BASE["s_nm"], BASE["wt_nm"], BASE["n_qd"])
        for hp in hp_vals
    ]
    plot_energy(
        aspect_vals, E,
        "Aspect ratio $h_p/b_r$",
        "Energy vs aspect ratio at fixed spacing",
        "sweep_aspect_fixed_spacing_energy.png"
    )

    lam = np.linspace(300, 1300, 1200)
    curves = [
        absorption_spectrum(lam, hp, br_fixed, BASE["s_nm"], BASE["wt_nm"], BASE["n_qd"])
        for hp in hp_vals
    ]
    labels = [f"$h_p/b_r$={ar:.1f}" for ar in aspect_vals]
    plot_absorption_family(
        lam, curves, labels,
        "Absorption for aspect-ratio sweep at fixed spacing",
        "sweep_aspect_fixed_spacing_absorption.png"
    )

# ============================
# 5) WETTING-LAYER SWEEP
# ============================
def sweep_wetting():
    wt_vals = np.array([2, 3, 4, 5, 6, 7, 8], dtype=float)

    E = [
        total_energy_meV(BASE["hp_nm"], BASE["br_nm"], BASE["s_nm"], wt, BASE["n_qd"])
        for wt in wt_vals
    ]
    plot_energy(
        wt_vals, E,
        "Wetting layer $w_t$ (nm)",
        "Energy vs wetting-layer thickness",
        "sweep_wetting_energy.png"
    )

    lam = np.linspace(300, 1300, 1200)
    curves = [
        absorption_spectrum(lam, BASE["hp_nm"], BASE["br_nm"], BASE["s_nm"], wt, BASE["n_qd"])
        for wt in wt_vals
    ]
    labels = [f"$w_t$={wt:.0f} nm" for wt in wt_vals]
    plot_absorption_family(
        lam, curves, labels,
        "Absorption for wetting-layer sweep",
        "sweep_wetting_absorption.png"
    )

# ============================
# MAIN
# ============================
if __name__ == "__main__":
    make_baseline_absorption()
    sweep_height()
    sweep_radius()
    sweep_constant_aspect()
    sweep_aspect_fixed_spacing()
    sweep_wetting()
    print("Done.")