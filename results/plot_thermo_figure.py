"""
Thermodynamics of RBM learning -- manuscript figure.

Reads the per-epoch thermodynamic traces produced by ``run_thermo_runs.py`` and
builds the three-panel figure

    (a)  <E>  vs epoch        expected energy
    (b)   S   vs epoch        thermodynamic entropy (k_B = 1)
    (c)   F   vs epoch        free energy, F = -(1/beta) log Z

Each curve is the mean over the independent seeds, with a shaded +/- 1 sigma band
(standard deviation across seeds at fixed epoch).

It also writes the single-panel "preliminary" versions of each quantity, a
delta_F diagnostic (is delta_F < 0 for most of training?), and a summary text
file containing the mean |F - (E - T*S)| consistency error.

Usage
-----
    python results/plot_thermo_figure.py
    python results/plot_thermo_figure.py --indir results/extended_10000ep \
                                         --outdir results/extended_10000ep/figures
"""

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


HERE = os.path.dirname(os.path.abspath(__file__))

# Colour slots 1-3 of the validated categorical palette (blue / orange / aqua).
# One series per panel, so colour only distinguishes panels, never encodes data.
C_ENERGY = "#2a78d6"
C_ENTROPY = "#eb6834"
C_FREE = "#1baf7a"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#dcdcd8"

PANELS = [
    ("E", r"expected energy  $\langle E \rangle$", C_ENERGY, "(a)"),
    ("S", r"entropy  $S$", C_ENTROPY, "(b)"),
    ("F", r"free energy  $F$", C_FREE, "(c)"),
]


def set_style():
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "axes.edgecolor": INK_MUTED,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "mathtext.fontset": "dejavusans",
    })


def load(indir):
    """Load the master file if present, else concatenate data/run_*.csv."""
    master = os.path.join(indir, "all_runs.csv")
    if os.path.exists(master):
        return pd.read_csv(master)
    files = sorted(glob.glob(os.path.join(indir, "data", "run_*.csv")))
    if not files:
        raise SystemExit("no data found in %s (expected all_runs.csv or data/run_*.csv)" % indir)
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def summarise(df):
    """Mean and std across seeds at each epoch."""
    g = df.groupby("epoch")
    out = g[["E", "S", "F", "Z", "delta_F"]].agg(["mean", "std"])
    out.columns = ["_".join(c) for c in out.columns]
    out["n_seeds"] = g["seed"].nunique()
    return out.reset_index()


def draw_panel(ax, stats, key, ylabel, color, tag, xlabel=True):
    x = stats["epoch"].values
    m = stats["%s_mean" % key].values
    s = stats["%s_std" % key].values
    ax.fill_between(x, m - s, m + s, color=color, alpha=0.20, linewidth=0,
                    label=r"$\pm 1\sigma$")
    ax.plot(x, m, color=color, linewidth=1.6, label="mean")
    ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel("epoch")
    ax.set_xlim(x.min(), x.max())
    ax.grid(True, axis="both", alpha=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.text(-0.02, 1.04, tag, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=11, fontweight="bold", color=INK)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--indir", default=HERE)
    p.add_argument("--outdir", default=None)
    p.add_argument("--prefix", default="", help="prefix for the output filenames")
    args = p.parse_args()
    outdir = args.outdir or os.path.join(args.indir, "figures")
    os.makedirs(outdir, exist_ok=True)
    set_style()

    df = load(args.indir)
    stats = summarise(df)
    n_seeds = int(df["seed"].nunique())
    n_epochs = int(df["epoch"].max())
    pre = args.prefix

    # ---------- three-panel manuscript figure --------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5), constrained_layout=True)
    for ax, (key, ylabel, color, tag) in zip(axes, PANELS):
        draw_panel(ax, stats, key, ylabel, color, tag)
    axes[0].legend(loc="best")
    fig.suptitle("RBM learning as thermodynamic relaxation "
                 "(2x2 BAS, $n_v=n_h=4$, $T=1$, CD-1, %d seeds)" % n_seeds,
                 fontsize=11)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, "%sfig_thermo_relaxation.%s" % (pre, ext)),
                    bbox_inches="tight")
    plt.close(fig)

    # ---------- single-quantity preliminary plots ----------------------------
    names = {"E": "energy", "S": "entropy", "F": "free_energy"}
    titles = {"E": "Expected energy vs epoch",
              "S": "Entropy vs epoch",
              "F": "Free energy vs epoch"}
    for key, ylabel, color, tag in PANELS:
        fig, ax = plt.subplots(figsize=(5.2, 3.6), constrained_layout=True)
        draw_panel(ax, stats, key, ylabel, color, "")
        ax.set_title("%s  (mean $\\pm 1\\sigma$, %d seeds)" % (titles[key], n_seeds))
        ax.legend(loc="best")
        fig.savefig(os.path.join(outdir, "%s%s_vs_epoch.png" % (pre, names[key])),
                    bbox_inches="tight")
        plt.close(fig)

    # ---------- delta_F diagnostic -------------------------------------------
    d = df.dropna(subset=["delta_F"])
    frac_neg_overall = float((d["delta_F"] < 0).mean())
    per_epoch_frac = d.groupby("epoch")["delta_F"].apply(lambda v: (v < 0).mean())

    # A single-epoch increment is dominated by CD sampling noise, so the sign of
    # delta_F on its own understates the drift.  Coarse-grain: what fraction of
    # W-epoch windows have F[n+W] - F[n] < 0?
    windows = [w for w in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000) if w < n_epochs]
    frac_by_window = []
    for w in windows:
        per_seed = []
        for _, g in df.groupby("seed"):
            F = g.sort_values("epoch")["F"].values
            per_seed.append(float((F[w:] - F[:-w] < 0).mean()))
        frac_by_window.append(float(np.mean(per_seed)))
    first_last = df.groupby("seed")["F"]
    frac_seeds_down = float((first_last.last() < first_last.first()).mean())

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.5), constrained_layout=True)
    ax = axes[0]
    ax.plot(per_epoch_frac.index.values, per_epoch_frac.values,
            color=C_FREE, linewidth=0.8, alpha=0.75)
    ax.axhline(0.5, color=INK_MUTED, linewidth=1.0, linestyle="--",
               label="0.5 (no drift)")
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"fraction of seeds with $\Delta F < 0$")
    ax.set_ylim(0, 1)
    ax.set_xlim(per_epoch_frac.index.min(), per_epoch_frac.index.max())
    ax.grid(True, alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="best")
    ax.text(-0.02, 1.04, "(a)", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=11, fontweight="bold")

    ax = axes[1]
    lim = float(np.nanpercentile(np.abs(d["delta_F"]), 99.5))
    ax.hist(d["delta_F"].clip(-lim, lim), bins=80, color=C_FREE, alpha=0.85)
    ax.axvline(0.0, color=INK_MUTED, linewidth=1.0, linestyle="--")
    ax.set_yscale("log")   # a converged run gives delta_F = 0 exactly; log keeps the tails readable
    ax.set_xlabel(r"$\Delta F = F[n+1] - F[n]$")
    ax.set_ylabel("count (log scale)")
    ax.grid(True, axis="y", alpha=0.7)
    ax.set_axisbelow(True)
    ax.text(-0.02, 1.04, "(b)", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=11, fontweight="bold")

    ax = axes[2]
    ax.semilogx(windows, frac_by_window, color=C_FREE, linewidth=1.6,
                marker="o", markersize=5)
    ax.axhline(0.5, color=INK_MUTED, linewidth=1.0, linestyle="--",
               label="0.5 (no drift)")
    ax.set_xlabel(r"coarse-graining window $W$ (epochs)")
    ax.set_ylabel(r"fraction with $F[n{+}W] - F[n] < 0$")
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right")
    ax.text(-0.02, 1.04, "(c)", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=11, fontweight="bold")

    for a in axes:
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
    fig.suptitle(r"Free-energy increments: %.1f%% of epoch steps have $\Delta F < 0$"
                 % (100 * frac_neg_overall), fontsize=11)
    fig.savefig(os.path.join(outdir, "%sdelta_F_diagnostic.png" % pre), bbox_inches="tight")
    plt.close(fig)

    # ---------- per-epoch summary csv + text summary -------------------------
    stats.to_csv(os.path.join(args.indir, "%sepoch_summary.csv" % pre), index=False)

    err = df["err"]
    lines = [
        "RBM thermodynamics -- summary",
        "=" * 46,
        "source directory      : %s" % os.path.abspath(args.indir),
        "seeds                 : %d" % n_seeds,
        "epochs per run        : %d" % n_epochs,
        "rows                  : %d" % len(df),
        "temperature           : %s (constant)" % np.unique(np.round(df['temperature'], 12)),
        "",
        "Consistency check  F  vs  E - T*S",
        "  mean |F - (E - T*S)| : %.6e" % err.mean(),
        "  max  |F - (E - T*S)| : %.6e" % err.max(),
        "  mean relative error  : %.6e" % (err / df["F"].abs()).mean(),
        "",
        "Free energy",
        "  F(epoch 1)   mean +/- sd : %+.4f +/- %.4f"
        % (stats["F_mean"].iloc[0], stats["F_std"].iloc[0]),
        "  F(epoch %d) mean +/- sd : %+.4f +/- %.4f"
        % (n_epochs, stats["F_mean"].iloc[-1], stats["F_std"].iloc[-1]),
        "  net change            : %+.4f" % (stats["F_mean"].iloc[-1] - stats["F_mean"].iloc[0]),
        "  fraction of epoch steps with delta_F < 0 : %.4f" % frac_neg_overall,
        "  mean delta_F                             : %+.6e" % d["delta_F"].mean(),
        "  fraction of seeds with F(last) < F(first): %.4f" % frac_seeds_down,
        "",
        "Coarse-grained free-energy decrease  (fraction of windows with F[n+W] - F[n] < 0)",
    ] + [
        "  W = %5d : %.4f" % (w, f) for w, f in zip(windows, frac_by_window)
    ] + [
        "",
        "Energy / entropy",
        "  <E>(1) -> <E>(%d) : %+.4f -> %+.4f" % (n_epochs, stats["E_mean"].iloc[0], stats["E_mean"].iloc[-1]),
        "  S(1)   -> S(%d)   : %+.4f -> %+.4f" % (n_epochs, stats["S_mean"].iloc[0], stats["S_mean"].iloc[-1]),
        "  ln(2^(n_v+n_h)) = ln(256) = %.4f  (maximum-entropy / uniform model)" % np.log(256),
    ]
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(args.indir, "%ssummary.txt" % pre), "w") as fh:
        fh.write(txt + "\n")
    print("\nfigures written to %s" % os.path.abspath(outdir))


if __name__ == "__main__":
    main()
