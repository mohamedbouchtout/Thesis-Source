"""
Thermodynamics of RBM learning -- second round of analyses.

Reads the per-epoch traces written by ``run_thermo_metrics.py`` and produces

  1. F_trajectories_500ep.png     F vs epoch, all 50 seeds (grey), ensemble mean
                                  (black), seeds 40 and 50 highlighted
     F_trajectories_5000ep.png    same for the 10 long runs
  2. Fmin_vs_Ffinal_500ep.png     (a) F_min vs F_final scatter
                                  (b) gap F_final - F_min against a noise-only null
                                  (c) epoch at which F_min is reached, against the null
  3. phase_F_vs_S_500ep.png       F vs S, points joined chronologically
     phase_F_vs_E_500ep.png       F vs E, points joined chronologically
     phase_space_5000ep.png       both planes for the long runs, coloured by epoch
  4. distributions_500ep.png      F, E, S across seeds at epochs 50/100/250/500
  5. long_runs_5000ep.png         E, S, F, KL, reconstruction error, data mass
  6. thermo_vs_learning.png       E, S, F, F_data against KL for both sweeps

plus tables/*.csv and summary.txt.

Why a null model for (2)
------------------------
F_min <= F_final for every run *by construction* (the minimum is taken over a
set that contains the final epoch), so every point sits on or above the
diagonal whatever the dynamics.  The informative quantities are how far above
(the gap) and when the minimum occurs.  The null is: ensemble-mean trend +
stationary AR(1) fluctuations, with the lag-1 autocorrelation and variance
fitted to each seed's own residual.  If observed gaps and minimum times look
like draws from that null, the "visits low-F states and leaves them" pattern is
what a noisy trajectory around a common trend produces anyway.

Usage
-----
    python plot_results.py --dir500 <dir>/runs_500ep --dir5000 <dir>/runs_5000ep --outdir <dir>
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

try:
    from scipy import stats as sps
except ImportError:  # pragma: no cover
    sps = None

ROOT = os.path.dirname(os.path.abspath(__file__))

# validated categorical palette, slots 1-3
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#dcdcd8"
LIGHT = "#b9b8b3"

HIGHLIGHT = [40, 50]                  # the two seeds singled out for inspection
SNAP_EPOCHS = [50, 100, 250, 500]
N_NULL = 2000
BAS_PATTERN_COUNT = 14
VISIBLE_DIM = 9
HIDDEN_DIM = 9
VISIBLE_STATE_COUNT = 2 ** VISIBLE_DIM
JOINT_STATE_COUNT = 2 ** (VISIBLE_DIM + HIDDEN_DIM)


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


def tag(ax, s):
    ax.text(-0.02, 1.04, s, transform=ax.transAxes, fontweight="bold",
            ha="right", va="bottom")


def save(fig, outdir, name):
    fig.savefig(os.path.join(outdir, name), bbox_inches="tight")
    plt.close(fig)
    print("  wrote", name)


def pivot(df, col):
    """epoch x seed table of one column."""
    return df.pivot(index="epoch", columns="seed", values=col).sort_index()


# --------------------------------------------------------------------------- 1
def fig_trajectories(df, outdir, name, highlight, title):
    P = pivot(df, "F")
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for s in P.columns:
        if s not in highlight:
            ax.plot(P.index, P[s], color=LIGHT, lw=0.6, alpha=0.55, zorder=1)
    for s, col in zip(highlight, (C1, C2, C3)):
        ax.plot(P.index, P[s], color=col, lw=1.3, zorder=3, label="seed %d" % s)
    ax.plot(P.index, P.mean(axis=1), color=INK, lw=2.4, zorder=4,
            label="ensemble mean (%d seeds)" % P.shape[1])
    ax.plot([], [], color=LIGHT, lw=1.2, label="individual seeds")
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"free energy  $F = -T\ln Z$")
    ax.set_title(title, loc="left", color=INK_MUTED, fontsize=10)
    ax.set_xlim(P.index.min(), P.index.max())
    ax.grid(True, axis="y")
    ax.legend(loc="best", ncol=2)
    save(fig, outdir, name)
    P.to_csv(os.path.join(outdir, "..", "tables", name.replace(".png", "_wide.csv")))


# --------------------------------------------------------------------------- 2
def ar1_null(P, n_null, rng):
    """Per seed: trend (ensemble mean) + AR(1) residual fitted to that seed.

    Returns dict seed -> (null_gap[n_null], null_argmin_epoch[n_null], phi).
    """
    trend = P.mean(axis=1).to_numpy()
    epochs = P.index.to_numpy()
    n = len(trend)
    out = {}
    for s in P.columns:
        r = P[s].to_numpy() - trend
        phi = float(np.corrcoef(r[:-1], r[1:])[0, 1])
        sd = float(r.std())
        sig = sd * np.sqrt(max(1.0 - phi ** 2, 1e-12))
        x = np.empty((n_null, n))
        x[:, 0] = r[0]
        eps = rng.normal(0.0, sig, size=(n_null, n))
        for t in range(1, n):
            x[:, t] = phi * x[:, t - 1] + eps[:, t]
        Fs = trend[None, :] + x
        gap = Fs[:, -1] - Fs.min(axis=1)
        am = epochs[np.argmin(Fs, axis=1)]
        out[s] = (gap, am, phi)
    return out


def fmin_table(df, null=None):
    P = pivot(df, "F")
    rows = []
    for s in P.columns:
        f = P[s]
        row = dict(seed=s, F_first=f.iloc[0], F_min=f.min(), epoch_of_min=int(f.idxmin()),
                   F_final=f.iloc[-1], F_max=f.max(), epoch_of_max=int(f.idxmax()),
                   gap=f.iloc[-1] - f.min())
        if null is not None:
            g, am, phi = null[s]
            row.update(ar1_phi=phi,
                       null_gap_median=float(np.median(g)),
                       null_gap_p05=float(np.quantile(g, 0.05)),
                       null_gap_p95=float(np.quantile(g, 0.95)),
                       gap_percentile_in_null=float(np.mean(g < row["gap"])),
                       null_epoch_of_min_median=float(np.median(am)))
        rows.append(row)
    return pd.DataFrame(rows)


def fig_fmin(df, tab, null, outdir, name):
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0),
                             gridspec_kw={"width_ratios": [1.15, 1, 1]})

    # (a) scatter
    ax = axes[0]
    sc = ax.scatter(tab.F_min, tab.F_final, c=tab.epoch_of_min, cmap="viridis",
                    s=28, edgecolor="white", linewidth=0.5, zorder=3,
                    vmin=df.epoch.min(), vmax=df.epoch.max())
    lo = min(tab.F_min.min(), tab.F_final.min()) - 0.05
    hi = max(tab.F_min.max(), tab.F_final.max()) + 0.05
    ax.plot([lo, hi], [lo, hi], color=INK_MUTED, lw=0.9, ls="--", zorder=1)
    ax.text(hi - 0.02, hi - 0.06, r"$F_{final}=F_{min}$", ha="right", va="top",
            color=INK_MUTED, fontsize=8, rotation=0)
    ax.fill_between([lo, hi], [lo, hi], lo, color="#f1f0ec", zorder=0)
    ax.text(hi - 0.02, lo + 0.03, "unreachable\n" r"($F_{min}\leq F_{final}$ always)",
            ha="right", va="bottom", color=INK_MUTED, fontsize=8)
    for s, col in zip(HIGHLIGHT, (C1, C2)):
        r = tab[tab.seed == s]
        if len(r):
            ax.scatter(r.F_min, r.F_final, s=70, facecolor="none", edgecolor=col,
                       linewidth=1.6, zorder=4)
            ax.annotate("seed %d" % s, (r.F_min.iloc[0], r.F_final.iloc[0]),
                        xytext=(6, 4), textcoords="offset points", color=col, fontsize=8)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$F_{min}$ = min over epochs")
    ax.set_ylabel(r"$F_{final}$ = $F$(epoch %d)" % df.epoch.max())
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("epoch at which $F_{min}$ is reached", fontsize=9)
    tag(ax, "(a)")

    # (b) gap vs null
    ax = axes[1]
    all_null_gap = np.concatenate([null[s][0] for s in null])
    bins = np.linspace(0, max(all_null_gap.max(), tab.gap.max()) * 1.02, 30)
    ax.hist(all_null_gap, bins=bins, density=True, color=LIGHT, alpha=0.8,
            label="null: trend + AR(1) noise")
    ax.hist(tab.gap, bins=bins, density=True, histtype="step", color=INK, lw=1.8,
            label="observed (50 seeds)")
    for s, col in zip(HIGHLIGHT, (C1, C2)):
        g = tab.loc[tab.seed == s, "gap"]
        if len(g):
            ax.axvline(g.iloc[0], color=col, lw=1.3, ls="--", label="seed %d" % s)
    ax.set_xlabel(r"gap  $F_{final} - F_{min}$")
    ax.set_ylabel("density")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.5)
    ax.legend(loc="upper right", fontsize=8)
    tag(ax, "(b)")

    # (c) epoch of minimum vs null
    ax = axes[2]
    all_null_am = np.concatenate([null[s][1] for s in null])
    ebins = np.linspace(df.epoch.min(), df.epoch.max(), 26)
    ax.hist(all_null_am, bins=ebins, density=True, color=LIGHT, alpha=0.8,
            label="null: trend + AR(1) noise")
    ax.hist(tab.epoch_of_min, bins=ebins, density=True, histtype="step", color=INK,
            lw=1.8, label="observed (50 seeds)")
    ax.set_xlabel(r"epoch at which $F_{min}$ is reached")
    ax.set_ylabel("density")
    ax.legend(loc="upper right")
    tag(ax, "(c)")

    fig.tight_layout()
    save(fig, outdir, name)


# --------------------------------------------------------------------------- 3
def pick_representative(df, exclude):
    """Seed whose F_final is closest to the ensemble median F_final."""
    fin = df[df.epoch == df.epoch.max()].set_index("seed").F
    fin = fin.drop([s for s in exclude if s in fin.index])
    return int((fin - fin.median()).abs().idxmin())


def fig_phase(df, xcol, xlabel, outdir, name, highlight):
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    for s, d in df.groupby("seed"):
        if s not in highlight:
            ax.plot(d[xcol], d.F, color=LIGHT, lw=0.45, alpha=0.45, zorder=1)
    m = df.groupby("epoch")[[xcol, "F"]].mean()
    ax.plot(m[xcol], m.F, color=INK, lw=2.0, zorder=4, label="ensemble mean path")
    labels = {highlight[0]: "seed %d" % highlight[0], highlight[1]: "seed %d" % highlight[1]}
    if len(highlight) > 2:
        labels[highlight[2]] = "seed %d (median $F_{final}$)" % highlight[2]
    for s, col in zip(highlight, (C1, C2, C3)):
        d = df[df.seed == s]
        ax.plot(d[xcol], d.F, color=col, lw=0.9, alpha=0.9, zorder=3, label=labels[s])
        ax.plot(d[xcol].iloc[0], d.F.iloc[0], "o", ms=6, mfc="white", mec=col, mew=1.5, zorder=5)
        ax.plot(d[xcol].iloc[-1], d.F.iloc[-1], "s", ms=6, color=col, mec="white", zorder=5)
    ax.plot([], [], "o", mfc="white", mec=INK_MUTED, label="start (epoch 1)")
    ax.plot([], [], "s", color=INK_MUTED, label="end (epoch %d)" % df.epoch.max())
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"free energy  $F$")
    ax.grid(True)
    ax.legend(loc="best", fontsize=8)
    save(fig, outdir, name)


def fig_phase_long(df, outdir, name):
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    norm = plt.Normalize(df.epoch.min(), df.epoch.max())
    for ax, xcol, xlabel, t in ((axes[0], "S", r"entropy  $S$", "(a)"),
                                (axes[1], "E", r"expected energy  $\langle E\rangle$", "(b)")):
        for s, d in df.groupby("seed"):
            pts = np.column_stack([d[xcol].to_numpy(), d.F.to_numpy()])
            segs = np.stack([pts[:-1], pts[1:]], axis=1)
            lc = LineCollection(segs, cmap="viridis", norm=norm, lw=1.0, alpha=0.9)
            lc.set_array(d.epoch.to_numpy()[:-1])
            ax.add_collection(lc)
            ax.plot(pts[0, 0], pts[0, 1], "o", ms=5, mfc="white", mec=INK_MUTED, zorder=5)
            ax.plot(pts[-1, 0], pts[-1, 1], "s", ms=5, color=INK, mec="white", zorder=5)
        ax.autoscale()
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"free energy  $F$")
        ax.grid(True)
        tag(ax, t)
    axes[0].plot([], [], "o", mfc="white", mec=INK_MUTED, label="start (epoch 1)")
    axes[0].plot([], [], "s", color=INK, label="end (epoch %d)" % df.epoch.max())
    axes[0].legend(loc="upper left")
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=axes,
                      fraction=0.025, pad=0.02)
    cb.set_label("epoch")
    save(fig, outdir, name)


# --------------------------------------------------------------------------- 4
def bimodality_coefficient(x):
    n = len(x)
    if sps is None or n < 4:
        return np.nan
    g = sps.skew(x, bias=False)
    k = sps.kurtosis(x, fisher=True, bias=False)
    return (g ** 2 + 1) / (k + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))


def dist_table(df, epochs, cols=("F", "E", "S", "KL")):
    rows = []
    for c in cols:
        for e in epochs:
            x = df.loc[df.epoch == e, c].to_numpy()
            row = dict(quantity=c, epoch=e, n=len(x), mean=x.mean(), std=x.std(ddof=1),
                       median=np.median(x), q25=np.quantile(x, .25), q75=np.quantile(x, .75),
                       min=x.min(), max=x.max())
            if sps is not None:
                row.update(skew=sps.skew(x, bias=False),
                           excess_kurtosis=sps.kurtosis(x, bias=False),
                           bimodality_coeff=bimodality_coefficient(x),
                           shapiro_p=sps.shapiro(x).pvalue)
            rows.append(row)
    return pd.DataFrame(rows)


def fig_distributions(df, outdir, name, epochs):
    panels = [("F", r"free energy  $F$", C3, "(a)"),
              ("E", r"expected energy  $\langle E\rangle$", C1, "(b)"),
              ("S", r"entropy  $S$", C2, "(c)")]
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.0))
    rng = np.random.default_rng(0)
    pos = np.arange(len(epochs))
    for ax, (c, lab, col, t) in zip(axes, panels):
        data = [df.loc[df.epoch == e, c].to_numpy() for e in epochs]
        vp = ax.violinplot(data, positions=pos, widths=0.8, showextrema=False)
        for b in vp["bodies"]:
            b.set_facecolor(col)
            b.set_edgecolor(col)
            b.set_alpha(0.25)
        ax.boxplot(data, positions=pos, widths=0.18, showfliers=False,
                   medianprops=dict(color=INK, lw=1.5),
                   boxprops=dict(color=INK_MUTED), whiskerprops=dict(color=INK_MUTED),
                   capprops=dict(color=INK_MUTED))
        for i, e in enumerate(epochs):
            d = df[df.epoch == e]
            jit = rng.uniform(-0.22, 0.22, len(d))
            ax.scatter(i + jit, d[c], s=7, color=INK_MUTED, alpha=0.6, lw=0, zorder=3)
            for s, hc in zip(HIGHLIGHT, (C1, C2)):
                v = d.loc[d.seed == s, c]
                if len(v):
                    ax.scatter(i + 0.3, v, s=34, marker="<", color=hc, zorder=4,
                               label=("seed %d" % s) if i == 0 else None)
        ax.set_xticks(pos)
        ax.set_xticklabels([str(e) for e in epochs])
        ax.set_xlabel("epoch")
        ax.set_ylabel(lab)
        ax.grid(True, axis="y")
        tag(ax, t)
    axes[0].legend(loc="best")
    fig.tight_layout()
    save(fig, outdir, name)


# --------------------------------------------------------------------------- 5
def fig_long(df, outdir, name):
    panels = [("E", r"$\langle E\rangle$"), ("S", r"$S$"), ("F", r"$F$"),
              ("KL", r"$D_{KL}(p_{data}\,\|\,p_{model})$"),
              ("recon_sq", "reconstruction error\n" r"$\langle (v-v')^2\rangle$"),
              ("data_mass", "model mass on the\n6 BAS patterns")]
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 6.6), sharex=True)
    for ax, (c, lab), t in zip(axes.ravel(), panels, "abcdef"):
        P = pivot(df, c)
        for s in P.columns:
            ax.plot(P.index, P[s], color=LIGHT, lw=0.7, alpha=0.8)
        ax.plot(P.index, P.mean(axis=1), color=INK, lw=2.0)
        ax.set_ylabel(lab)
        ax.grid(True, axis="y")
        tag(ax, "(%s)" % t)
    for ax in axes[1]:
        ax.set_xlabel("epoch")
    axes[0, 0].plot([], [], color=LIGHT, label="individual seeds (%d)" % df.seed.nunique())
    axes[0, 0].plot([], [], color=INK, lw=2, label="mean")
    axes[0, 0].legend(loc="lower left")
    fig.tight_layout()
    save(fig, outdir, name)


def settling_table(df, windows=(1000,)):
    """Per seed: linear drift of F over the last W epochs vs residual scatter."""
    rows = []
    for s, d in df.groupby("seed"):
        for W in windows:
            tail = d[d.epoch > d.epoch.max() - W]
            slope, icpt = np.polyfit(tail.epoch, tail.F, 1)
            resid = tail.F - (slope * tail.epoch + icpt)
            rows.append(dict(seed=s, window=W, F_final=d.F.iloc[-1],
                             slope_per_1000ep=slope * 1000, drift_over_window=slope * W,
                             resid_std=resid.std(), KL_final=d.KL.iloc[-1],
                             KL_min=d.KL.min()))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- 6
def corr_table(df, label):
    rows = []
    thermo = ["E", "S", "F", "F_data"]
    learn = ["KL", "recon_sq", "recon_ce"]
    diffs = df.sort_values(["seed", "epoch"]).groupby("seed")[thermo + learn].diff().dropna()
    for a in thermo:
        for b in learn:
            if sps is not None:
                rho_lvl = sps.spearmanr(df[a], df[b]).statistic
                rho_dif = sps.spearmanr(diffs[a], diffs[b]).statistic
                per_seed = df.groupby("seed").apply(
                    lambda d: sps.spearmanr(d[a], d[b]).statistic)
            else:
                rho_lvl = df[[a, b]].rank().corr().iloc[0, 1]
                rho_dif = diffs[[a, b]].rank().corr().iloc[0, 1]
                per_seed = df.groupby("seed").apply(lambda d: d[[a, b]].rank().corr().iloc[0, 1])
            rows.append(dict(sweep=label, thermo=a, learning=b,
                             spearman_pooled_levels=rho_lvl,
                             spearman_within_seed_median=float(np.median(per_seed)),
                             spearman_increments=rho_dif))
    return pd.DataFrame(rows)


def fig_thermo_learning(d500, d5000, outdir, name):
    cols = [("F", r"$F$"), ("F_data", r"$\langle F(v)\rangle_{data}$"),
            ("E", r"$\langle E\rangle$"), ("S", r"$S$")]
    fig, axes = plt.subplots(2, 4, figsize=(15.0, 7.6), layout="constrained")
    for row, (df, lab, stride) in enumerate(((d500, "50 seeds x 500 epochs", 5),
                                             (d5000, "10 seeds x 5000 epochs", 10))):
        sub = df[df.epoch % stride == 0]
        norm = plt.Normalize(df.epoch.min(), df.epoch.max())
        for ax, (c, clab) in zip(axes[row], cols):
            sc = ax.scatter(sub.KL, sub[c], c=sub.epoch, cmap="viridis", norm=norm,
                            s=4, alpha=0.6, lw=0, rasterized=True)
            ax.set_xlabel(r"$D_{KL}(p_{data}\,\|\,p_{model})$")
            ax.set_ylabel(clab)
            ax.grid(True)
        axes[row, 0].set_title(lab, loc="left", color=INK_MUTED, fontsize=10)
        cb = fig.colorbar(sc, ax=axes[row], fraction=0.02, pad=0.01)
        cb.set_label("epoch")
    save(fig, outdir, name)


# --------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir500")
    p.add_argument("--dir5000")
    p.add_argument("--outdir")
    p.add_argument("--visible-dim", type=int, default=9)
    p.add_argument("--hidden-dim", type=int, default=9)
    args = p.parse_args()

    BAS_PATTERN_COUNT = (args.visible_dim + args.hidden_dim) - 2
    VISIBLE_DIM = args.visible_dim
    HIDDEN_DIM = args.hidden_dim
    VISIBLE_STATE_COUNT = 2 ** VISIBLE_DIM
    JOINT_STATE_COUNT = 2 ** (VISIBLE_DIM + HIDDEN_DIM)

    set_style()
    figdir = os.path.join(args.outdir, "figures")
    tabdir = os.path.join(args.outdir, "tables")
    os.makedirs(figdir, exist_ok=True)
    os.makedirs(tabdir, exist_ok=True)

    d500 = pd.read_csv(os.path.join(args.dir500, "all_runs.csv"))
    d5000 = pd.read_csv(os.path.join(args.dir5000, "all_runs.csv"))
    rng = np.random.default_rng(12345)
    lines = []

    # 1 ----------------------------------------------------------------------
    fig_trajectories(d500, figdir, "F_trajectories_500ep.png", HIGHLIGHT,
                     "50 seeds, 500 epochs (full batch: 1 CD-1 update / epoch)")
    fig_trajectories(d5000, figdir, "F_trajectories_5000ep.png", [],
                     "10 seeds, 5000 epochs")

    # 2 ----------------------------------------------------------------------
    P = pivot(d500, "F")
    null = ar1_null(P, N_NULL, rng)
    tab = fmin_table(d500, null)
    tab.to_csv(os.path.join(tabdir, "Fmin_Ffinal_500ep.csv"), index=False)
    fmin_table(d5000).to_csv(os.path.join(tabdir, "Fmin_Ffinal_5000ep.csv"), index=False)
    fig_fmin(d500, tab, null, figdir, "Fmin_vs_Ffinal_500ep.png")

    all_null_gap = np.concatenate([null[s][0] for s in null])
    all_null_am = np.concatenate([null[s][1] for s in null])
    pct = tab.gap_percentile_in_null.to_numpy()
    lines += ["F_min vs F_final  (50 seeds x 500 epochs)",
              "-" * 60,
              "F_min <= F_final holds for every seed by construction.",
              "gap F_final - F_min   observed: median %.3f  mean %.3f  [%.3f, %.3f]"
              % (tab.gap.median(), tab.gap.mean(), tab.gap.min(), tab.gap.max()),
              "                      null    : median %.3f  mean %.3f"
              % (np.median(all_null_gap), all_null_gap.mean()),
              "epoch of F_min        observed: median %.0f   null: median %.0f"
              % (tab.epoch_of_min.median(), np.median(all_null_am)),
              "F_min at epoch <= 10  : %d / %d seeds" % ((tab.epoch_of_min <= 10).sum(), len(tab)),
              "F_final < F_first     : %d / %d seeds" % ((tab.F_final < tab.F_first).sum(), len(tab)),
              "per-seed AR(1) phi    : median %.3f  [%.3f, %.3f]"
              % (tab.ar1_phi.median(), tab.ar1_phi.min(), tab.ar1_phi.max()),
              "gap percentile in own null: >0.95 for %d seeds, <0.05 for %d seeds "
              "(expected ~2.5 each under the null)" % ((pct > 0.95).sum(), (pct < 0.05).sum())]
    if sps is not None:
        ks = sps.kstest(pct, "uniform")
        lines.append("KS test percentiles vs Uniform(0,1): D = %.3f, p = %.3f" % (ks.statistic, ks.pvalue))
    for s in HIGHLIGHT:
        r = tab[tab.seed == s].iloc[0]
        lines.append("seed %d: F_first %.3f  F_min %.3f @ epoch %d  F_final %.3f  gap %.3f  "
                     "(null 90%% range %.3f-%.3f, percentile %.2f)"
                     % (s, r.F_first, r.F_min, r.epoch_of_min, r.F_final, r.gap,
                        r.null_gap_p05, r.null_gap_p95, r.gap_percentile_in_null))
    lines.append("")

    # 3 ----------------------------------------------------------------------
    rep = pick_representative(d500, HIGHLIGHT)
    hl3 = HIGHLIGHT + [rep]
    fig_phase(d500, "S", r"entropy  $S$", figdir, "phase_F_vs_S_500ep.png", hl3)
    fig_phase(d500, "E", r"expected energy  $\langle E\rangle$", figdir,
              "phase_F_vs_E_500ep.png", hl3)
    fig_phase_long(d5000, figdir, "phase_space_5000ep.png")
    ranges = d500.agg({"E": ["min", "max"], "S": ["min", "max"], "F": ["min", "max"]})
    lines += [    "Phase space (500 epochs): E in [%.3f, %.3f], S in [%.4f, %.4f] (ln %d = %.4f), "
              "F in [%.3f, %.3f]" % (ranges.E["min"], ranges.E["max"], ranges.S["min"],
                 ranges.S["max"], JOINT_STATE_COUNT, np.log(JOINT_STATE_COUNT),
                           ranges.F["min"], ranges.F["max"]),
              "representative (median F_final) seed: %d" % rep, ""]

    # 4 ----------------------------------------------------------------------
    fig_distributions(d500, figdir, "distributions_500ep.png", SNAP_EPOCHS)
    dt = dist_table(d500, SNAP_EPOCHS)
    dt.to_csv(os.path.join(tabdir, "distributions_500ep.csv"), index=False)
    lines.append("Distribution of F across seeds")
    for _, r in dt[dt.quantity == "F"].iterrows():
        extra = ""
        if "bimodality_coeff" in r:
            extra = "  skew %+.2f  BC %.3f  Shapiro p %.3f" % (r["skew"], r["bimodality_coeff"], r["shapiro_p"])
        lines.append("  epoch %4d: mean %.3f  std %.3f  [%.3f, %.3f]%s"
                     % (r.epoch, r["mean"], r["std"], r["min"], r["max"], extra))
    lines += ["  (BC > 0.555 would suggest bimodality)", ""]

    # 5 ----------------------------------------------------------------------
    fig_long(d5000, figdir, "long_runs_5000ep.png")
    st = settling_table(d5000, windows=(1000, 500))
    st.to_csv(os.path.join(tabdir, "settling_5000ep.csv"), index=False)
    s1000 = st[st.window == 1000]
    lines += ["Long runs (10 seeds x 5000 epochs)",
              "-" * 60,
              "F: epoch 1 %.3f +/- %.3f  ->  epoch 5000 %.3f +/- %.3f"
              % (d5000[d5000.epoch == 1].F.mean(), d5000[d5000.epoch == 1].F.std(),
                 d5000[d5000.epoch == 5000].F.mean(), d5000[d5000.epoch == 5000].F.std()),
              "KL: epoch 1 %.3f  ->  epoch 5000 %.3f +/- %.3f   (uniform model: ln(%d/%d) = %.3f)"
              % (d5000[d5000.epoch == 1].KL.mean(), d5000[d5000.epoch == 5000].KL.mean(),
                 d5000[d5000.epoch == 5000].KL.std(), VISIBLE_STATE_COUNT, BAS_PATTERN_COUNT,
                 np.log(VISIBLE_STATE_COUNT / BAS_PATTERN_COUNT)),
              "F drift over last 1000 epochs: median %.3f  (per-seed residual std median %.3f)"
              % (s1000.drift_over_window.median(), s1000.resid_std.median()),
              "seeds still drifting (|drift| > 3 x resid std): %d / %d"
              % ((s1000.drift_over_window.abs() > 3 * s1000.resid_std).sum(), len(s1000))]

    ext = os.path.join(ROOT, "results", "extended_10000ep", "all_runs.csv")
    if os.path.exists(ext):
        e10 = pd.read_csv(ext, usecols=["seed", "epoch", "F"])
        e10 = e10[e10.seed.isin(d5000.seed.unique())]
        both = d5000[["seed", "epoch", "F"]].merge(e10, on=["seed", "epoch"], suffixes=("", "_10k"))
        lines.append("identical to results/extended_10000ep over epochs 1-5000: max |dF| = %.2e"
                     % (both.F - both.F_10k).abs().max())
        for e in (5000, 7500, 10000):
            x = e10[e10.epoch == e].F
            lines.append("  same seeds in the 10k run, epoch %5d: F = %.3f +/- %.3f" % (e, x.mean(), x.std()))
    lines.append("")

    # 6 ----------------------------------------------------------------------
    fig_thermo_learning(d500, d5000, figdir, "thermo_vs_learning.png")
    ct = pd.concat([corr_table(d500, "500ep"), corr_table(d5000, "5000ep")], ignore_index=True)
    ct.to_csv(os.path.join(tabdir, "correlations.csv"), index=False)
    lines.append("Spearman correlation with KL  (pooled levels | within-seed median | increments)")
    for _, r in ct[ct.learning == "KL"].iterrows():
        lines.append("  %-6s %-7s %+.3f | %+.3f | %+.3f"
                     % (r.sweep, r.thermo, r.spearman_pooled_levels,
                        r.spearman_within_seed_median, r.spearman_increments))
    lines.append("Identity: KL = beta*(F_data - F) - ln 14, so F enters the learning objective "
                 "with a MINUS sign.")

    txt = "\n".join(lines)
    print("\n" + txt)
    with open(os.path.join(args.outdir, "summary.txt"), "w") as fh:
        fh.write(txt + "\n")


if __name__ == "__main__":
    main()
