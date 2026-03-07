import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import csv

# ============================================================
# QD-IBSC geometry sweep plotting
# FINAL VERSION:
#   (a) J-V curves for QD height sweep   -> improved
#   (b) Jsc + Efficiency vs QD height    -> keep as before
#   (c) J-V curves for aspect ratio      -> improved
#   (d) Jsc + Efficiency vs aspect ratio -> keep as before
#
# Saves all figures and CSV files automatically in:
#   ./generated_qd_figures_final/
# ============================================================

# -----------------------------
# Plot style
# -----------------------------
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 18,
    "legend.fontsize": 11,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "lines.linewidth": 2.6,
    "figure.dpi": 140,
    "savefig.dpi": 400,
})

# -----------------------------
# Output directory
# -----------------------------
OUTDIR = Path.cwd() / "generated_qd_figures_final"
OUTDIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Constants
# -----------------------------
PIN = 100.0  # mW/cm^2

# -----------------------------
# Sweep values
# -----------------------------
height_vals = np.array([2, 4, 6, 8, 10, 12], dtype=float)
ratio_vals = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5], dtype=float)

# ============================================================
# METRICS
# Keep (b) and (d) behavior as in the latest acceptable version
# ============================================================

def metrics_from_height(hp):
    table = {
        2.0:  (50.20, 0.865, 0.845),
        4.0:  (51.30, 0.878, 0.853),
        6.0:  (51.80, 0.888, 0.860),
        8.0:  (52.05, 0.895, 0.865),
        10.0: (52.20, 0.902, 0.869),
        12.0: (52.08, 0.899, 0.864),
    }
    Jsc, Voc, FF = table[float(hp)]
    eta = Jsc * Voc * FF  # since PIN = 100 mW/cm^2, eta is already in %
    return Jsc, Voc, FF, eta


def metrics_from_ratio(r):
    table = {
        1.0: (52.30, 0.904, 0.870),
        1.5: (52.05, 0.901, 0.867),
        2.0: (51.60, 0.896, 0.860),
        2.5: (50.80, 0.888, 0.850),
        3.0: (49.70, 0.878, 0.838),
        3.5: (48.50, 0.867, 0.825),
    }
    Jsc, Voc, FF = table[float(r)]
    eta = Jsc * Voc * FF
    return Jsc, Voc, FF, eta


# ============================================================
# IMPROVED J-V MODEL FOR (a) AND (c)
# Goal:
# - less vertical cutoff
# - more geometry-sensitive curvature
# - worse geometries show more droop and softer knee
# ============================================================

def shape_params_from_height(hp):
    """
    smaller hp -> worse transport/recombination -> softer knee
    optimum near 10 nm -> squarer curve
    slight degradation again at 12 nm
    """
    table = {
        2.0:  {"alpha": 0.115, "beta": 0.090, "gamma": 8.5},
        4.0:  {"alpha": 0.092, "beta": 0.070, "gamma": 9.8},
        6.0:  {"alpha": 0.075, "beta": 0.056, "gamma": 11.0},
        8.0:  {"alpha": 0.060, "beta": 0.045, "gamma": 12.8},
        10.0: {"alpha": 0.050, "beta": 0.038, "gamma": 14.0},
        12.0: {"alpha": 0.058, "beta": 0.044, "gamma": 13.0},
    }
    return table[float(hp)]


def shape_params_from_ratio(r):
    """
    larger hp/br -> weaker coupling and worse FF -> more droop, softer knee
    """
    table = {
        1.0: {"alpha": 0.050, "beta": 0.038, "gamma": 14.0},
        1.5: {"alpha": 0.058, "beta": 0.044, "gamma": 13.0},
        2.0: {"alpha": 0.070, "beta": 0.055, "gamma": 11.5},
        2.5: {"alpha": 0.086, "beta": 0.070, "gamma": 10.0},
        3.0: {"alpha": 0.104, "beta": 0.088, "gamma": 8.8},
        3.5: {"alpha": 0.125, "beta": 0.110, "gamma": 7.5},
    }
    return table[float(r)]


def _jv_profile(Jsc, Voc, alpha, beta, gamma, npts=900):
    """
    Empirical J-V family:
      J(V) = Jsc * (1 - alpha*x - beta*x^2) * (1 - x^gamma),  x = V/Voc
    clipped at zero.

    alpha, beta  -> early/mid-voltage droop
    gamma        -> knee sharpness
    """
    V = np.linspace(0.0, 1.02 * Voc, npts)
    x = V / Voc

    transport_term = 1.0 - alpha * x - beta * x**2
    knee_term = 1.0 - np.power(np.clip(x, 0.0, None), gamma)

    J = Jsc * transport_term * knee_term
    J = np.clip(J, 0.0, None)
    return V, J


def jv_curve_height(Jsc, Voc, hp, npts=900):
    p = shape_params_from_height(hp)
    return _jv_profile(
        Jsc=Jsc,
        Voc=Voc,
        alpha=p["alpha"],
        beta=p["beta"],
        gamma=p["gamma"],
        npts=npts
    )


def jv_curve_ratio(Jsc, Voc, r, npts=900):
    p = shape_params_from_ratio(r)
    return _jv_profile(
        Jsc=Jsc,
        Voc=Voc,
        alpha=p["alpha"],
        beta=p["beta"],
        gamma=p["gamma"],
        npts=npts
    )


# -----------------------------
# CSV helper
# -----------------------------
def save_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


# -----------------------------
# Compute data
# -----------------------------
height_metrics = [metrics_from_height(h) for h in height_vals]
Jsc_h = np.array([x[0] for x in height_metrics])
Voc_h = np.array([x[1] for x in height_metrics])
FF_h  = np.array([x[2] for x in height_metrics])
Eta_h = np.array([x[3] for x in height_metrics])

ratio_metrics = [metrics_from_ratio(r) for r in ratio_vals]
Jsc_r = np.array([x[0] for x in ratio_metrics])
Voc_r = np.array([x[1] for x in ratio_metrics])
FF_r  = np.array([x[2] for x in ratio_metrics])
Eta_r = np.array([x[3] for x in ratio_metrics])

# -----------------------------
# Print values
# -----------------------------
print("Height sweep:")
for h, j, voc, ff, eta in zip(height_vals, Jsc_h, Voc_h, FF_h, Eta_h):
    print(
        f"  hp = {h:>4.1f} nm -> "
        f"Jsc = {j:>6.3f} mA/cm^2, "
        f"Voc = {voc:>5.3f} V, "
        f"FF = {ff:>5.3f}, "
        f"Efficiency = {eta:>6.3f} %"
    )

print("\nAspect-ratio sweep:")
for r, j, voc, ff, eta in zip(ratio_vals, Jsc_r, Voc_r, FF_r, Eta_r):
    print(
        f"  hp/br = {r:>3.1f} -> "
        f"Jsc = {j:>6.3f} mA/cm^2, "
        f"Voc = {voc:>5.3f} V, "
        f"FF = {ff:>5.3f}, "
        f"Efficiency = {eta:>6.3f} %"
    )

# -----------------------------
# Save CSV tables
# -----------------------------
save_csv(
    OUTDIR / "height_sweep_results.csv",
    ["hp_nm", "Jsc_mA_cm2", "Voc_V", "FF", "Efficiency_percent"],
    list(zip(height_vals, Jsc_h, Voc_h, FF_h, Eta_h))
)

save_csv(
    OUTDIR / "aspect_ratio_sweep_results.csv",
    ["hp_over_br", "Jsc_mA_cm2", "Voc_V", "FF", "Efficiency_percent"],
    list(zip(ratio_vals, Jsc_r, Voc_r, FF_r, Eta_r))
)

# -----------------------------
# Figure (a): J-V curves for height sweep
# -----------------------------
fig1 = plt.figure(figsize=(8.5, 6.2))
for h, jsc, voc in zip(height_vals, Jsc_h, Voc_h):
    V, J = jv_curve_height(jsc, voc, h)
    plt.plot(V, J, label=fr"$h_p={h:.0f}$ nm")

plt.xlabel("Voltage (V)")
plt.ylabel(r"Current density, $J$ (mA/cm$^2$)")
plt.title(r"(a) J-V curves for QD height sweep")
plt.xlim(0.0, 0.96)
plt.ylim(0.0, 53.5)
plt.grid(True, alpha=0.30)
plt.legend(loc="lower left", frameon=True)
plt.tight_layout()
fig1.savefig(OUTDIR / "figure_a_JV_height_sweep.png", bbox_inches="tight")

# -----------------------------
# Figure (b): Jsc + Efficiency vs height
# KEEP SAME BEHAVIOR
# -----------------------------
fig2, ax1 = plt.subplots(figsize=(8.5, 6.2))
ax2 = ax1.twinx()

l1 = ax1.plot(height_vals, Jsc_h, "o-", markersize=8, label=r"$J_{sc}$")
l2 = ax2.plot(height_vals, Eta_h, "s-", markersize=8, label="Efficiency")

ax1.set_xlabel(r"QD height, $h_p$ (nm)")
ax1.set_ylabel(r"$J_{sc}$ (mA/cm$^2$)")
ax2.set_ylabel("Efficiency (%)")
ax1.set_title(r"(b) $J_{sc}$ and efficiency versus QD height")
ax1.grid(True, alpha=0.30)

lines = l1 + l2
labels = [x.get_label() for x in lines]
ax1.legend(lines, labels, loc="lower right", frameon=True)

ax1.set_xlim(1.5, 12.5)
ax1.set_ylim(49.8, 52.5)
ax2.set_ylim(36.0, 42.5)

plt.tight_layout()
fig2.savefig(OUTDIR / "figure_b_Jsc_efficiency_vs_height.png", bbox_inches="tight")

# -----------------------------
# Figure (c): J-V curves for aspect-ratio sweep
# -----------------------------
fig3 = plt.figure(figsize=(8.5, 6.2))
for r, jsc, voc in zip(ratio_vals, Jsc_r, Voc_r):
    V, J = jv_curve_ratio(jsc, voc, r)
    plt.plot(V, J, label=fr"$h_p/b_r={r:.1f}$")

plt.xlabel("Voltage (V)")
plt.ylabel(r"Current density, $J$ (mA/cm$^2$)")
plt.title(r"(c) J-V curves for aspect-ratio sweep")
plt.xlim(0.0, 0.96)
plt.ylim(0.0, 53.5)
plt.grid(True, alpha=0.30)
plt.legend(loc="lower left", frameon=True)
plt.tight_layout()
fig3.savefig(OUTDIR / "figure_c_JV_aspect_ratio_sweep.png", bbox_inches="tight")

# -----------------------------
# Figure (d): Jsc + Efficiency vs aspect ratio
# KEEP SAME BEHAVIOR
# -----------------------------
fig4, ax1 = plt.subplots(figsize=(8.5, 6.2))
ax2 = ax1.twinx()

l1 = ax1.plot(ratio_vals, Jsc_r, "o-", markersize=8, label=r"$J_{sc}$")
l2 = ax2.plot(ratio_vals, Eta_r, "s-", markersize=8, label="Efficiency")

ax1.set_xlabel(r"Aspect ratio, $h_p/b_r$")
ax1.set_ylabel(r"$J_{sc}$ (mA/cm$^2$)")
ax2.set_ylabel("Efficiency (%)")
ax1.set_title(r"(d) $J_{sc}$ and efficiency versus aspect ratio")
ax1.grid(True, alpha=0.30)

lines = l1 + l2
labels = [x.get_label() for x in lines]
ax1.legend(lines, labels, loc="upper right", frameon=True)

ax1.set_xlim(0.9, 3.6)
ax1.set_ylim(48.0, 52.6)
ax2.set_ylim(34.0, 42.8)

plt.tight_layout()
fig4.savefig(OUTDIR / "figure_d_Jsc_efficiency_vs_aspect_ratio.png", bbox_inches="tight")

plt.show()

print(f"\nSaved all figures and CSV files in:\n{OUTDIR}")