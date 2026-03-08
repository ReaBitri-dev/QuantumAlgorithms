import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import csv

# ============================================================
# QD-IBSC geometry sweep plotting
# Full corrected script
#
# Saves:
#   (a) J-V curves for QD height sweep
#   (b) Jsc + Efficiency versus QD height
#   (c) J-V curves for aspect-ratio sweep
#   (d) Jsc + Efficiency versus aspect ratio
#
# Output folder:
#   ./generated_qd_figures_final_fixed_a/
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
OUTDIR = Path.cwd() / "generated_qd_figures_final_fixed_a"
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
# DEVICE METRICS
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
    eta = Jsc * Voc * FF  # since PIN = 100 mW/cm^2
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
# J-V MODELS
# Figure (a) fixed again
# Figure (c) kept from improved version
# ============================================================

def shape_params_from_height(hp):
    """
    Height sweep:
    low hp is worse, but not overly degraded across the full voltage range.
    10 nm remains best, 12 nm slightly below 10 nm.
    """
    table = {
        2.0:  {"alpha": 0.028, "beta": 0.016, "gamma": 8.8,  "delta": 10.5, "lam": 0.18},
        4.0:  {"alpha": 0.024, "beta": 0.014, "gamma": 9.8,  "delta": 10.8, "lam": 0.15},
        6.0:  {"alpha": 0.021, "beta": 0.012, "gamma": 10.8, "delta": 11.2, "lam": 0.12},
        8.0:  {"alpha": 0.018, "beta": 0.010, "gamma": 11.8, "delta": 11.6, "lam": 0.10},
        10.0: {"alpha": 0.016, "beta": 0.009, "gamma": 12.6, "delta": 12.0, "lam": 0.08},
        12.0: {"alpha": 0.017, "beta": 0.0095,"gamma": 12.1, "delta": 11.8, "lam": 0.09},
    }
    return table[float(hp)]


def shape_params_from_ratio(r):
    table = {
        1.0: {"alpha": 0.030, "beta": 0.018, "gamma": 12.8, "delta": 9.8},
        1.5: {"alpha": 0.036, "beta": 0.022, "gamma": 12.0, "delta": 9.2},
        2.0: {"alpha": 0.046, "beta": 0.029, "gamma": 10.8, "delta": 8.4},
        2.5: {"alpha": 0.060, "beta": 0.040, "gamma": 9.5,  "delta": 7.6},
        3.0: {"alpha": 0.078, "beta": 0.055, "gamma": 8.3,  "delta": 6.8},
        3.5: {"alpha": 0.100, "beta": 0.074, "gamma": 7.2,  "delta": 6.0},
    }
    return table[float(r)]


def jv_curve_height(Jsc, Voc, hp, npts=900):
    """
    Height-sweep J-V with degradation concentrated near the knee,
    not across the whole voltage range.
    """
    p = shape_params_from_height(hp)

    V = np.linspace(0.0, 1.02 * Voc, npts)
    x = np.clip(V / Voc, 0.0, None)

    # mild transport droop
    base = 1.0 - p["alpha"] * x - p["beta"] * x**2

    # main knee
    knee = 1.0 - x**p["gamma"]

    # extra late-voltage degradation only near high bias
    late = 1.0 - p["lam"] * x**p["delta"]

    J = Jsc * base * knee * late
    J = np.clip(J, 0.0, None)
    return V, J


def _jv_profile_ratio(Jsc, Voc, alpha, beta, gamma, delta, npts=900):
    V = np.linspace(0.0, 1.02 * Voc, npts)
    x = np.clip(V / Voc, 0.0, None)

    transport = 1.0 - alpha * x - beta * x**2
    knee = 1.0 - x**gamma
    late = 1.0 / (1.0 + x**delta)

    J = Jsc * transport * knee * (0.55 + 0.45 * late)
    J = np.clip(J, 0.0, None)
    return V, J


def jv_curve_ratio(Jsc, Voc, r, npts=900):
    p = shape_params_from_ratio(r)
    return _jv_profile_ratio(
        Jsc=Jsc,
        Voc=Voc,
        alpha=p["alpha"],
        beta=p["beta"],
        gamma=p["gamma"],
        delta=p["delta"],
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