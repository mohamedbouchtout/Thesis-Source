# RBM learning: individual trajectories, phase space and learning metrics

Second round of analyses. Same model, data and hyper-parameters as `results/`:

| setting | value |
|---|---|
| dataset | 2×2 Bars-and-Stripes (6 patterns) |
| `visible_dim` / `hidden_dim` | 4 / 4 |
| `T` | 1 (`annealing_decay = 0`) |
| training | CD-1, full batch (1 update / epoch), lr 0.15 |
| seeds | 1–50 for the 500-epoch sweep, 1–10 for the 5000-epoch sweep |

The 50 × 500 runs were regenerated with the new driver. **They are bit-identical to
`results/all_runs.csv`** (max |ΔF| = 0.0), because the new metrics are computed
exactly by enumeration and draw no random numbers. The 10 × 5000 runs are
bit-identical to the first 5000 epochs of seeds 1–10 in `results/extended_10000ep/`.

## Files

```
results-2/
  run_thermo_metrics.py        driver (thermodynamics + learning metrics, every epoch)
  plot_results2.py             all figures, tables and summary.txt
  summary.txt                  key numbers quoted below
  runs_500ep/                  50 seeds x 500 epochs   (all_runs.csv, data/run_001..050.csv)
  runs_5000ep/                 10 seeds x 5000 epochs  (all_runs.csv, data/run_001..010.csv)
  figures/
    F_trajectories_500ep.png   task 1: all 50 seeds, mean, seeds 40 & 50
    F_trajectories_5000ep.png  task 1 for the long runs
    Fmin_vs_Ffinal_500ep.png   task 2: scatter + null-model test
    phase_F_vs_S_500ep.png     task 3A
    phase_F_vs_E_500ep.png     task 3B
    phase_space_5000ep.png     task 3, long runs, coloured by epoch
    distributions_500ep.png    task 4: F, E, S at epochs 50/100/250/500
    long_runs_5000ep.png       task 5 + 6: E, S, F, KL, reconstruction error, data mass
    thermo_vs_learning.png     task 6: E, S, F, <F(v)>_data against KL
  tables/
    F_trajectories_*_wide.csv  epoch x seed matrix of F
    Fmin_Ffinal_500ep.csv      per seed: F_first, F_min, epoch of min, F_final, gap, null stats
    Fmin_Ffinal_5000ep.csv     same, long runs (no null)
    distributions_500ep.csv    mean/std/quantiles/skew/bimodality/Shapiro per epoch and quantity
    settling_5000ep.csv        drift of F over the last 1000 / 500 epochs vs residual scatter
    correlations.csv           Spearman: {E,S,F,F_data} x {KL, recon_sq, recon_ce}
```

### CSV columns (both sweeps)

| column | meaning |
|---|---|
| `Z, E, S, F, F_check, err, delta_F` | as in `results/`: `F = -T ln Z`, `F_check = E - T S` |
| `F_data` | mean visible free energy of the 6 patterns, `F(v) = -T ln Σ_h e^{-E(v,h)/T}` |
| `KL` | `D_KL(p_data ‖ p_model)`, exact. Same definition as `RBM.KL_div_BIG_SMALL`, and checked against it for every seed |
| `NLL` | `-(1/6) Σ_d ln p_model(d)` = `KL + ln 6` |
| `data_mass` | model probability on the 6 BAS patterns (1 = perfect, 6/16 = 0.375 = uniform) |
| `recon_sq` | exact expectation of `RBM.average_squared_error` (v → h → v′), per unit |
| `recon_ce` | exact expectation of `RBM.recon_c_e` (cross-entropy of v against p(v′\|h)) |

The two reconstruction metrics are the *expected values* of the sampled
routines in the RBM class, summed over all 16 hidden states. A Monte-Carlo check
of 20 000 calls to each class routine agrees to 3 decimals.

### Consistency

| sweep | mean \|F − (E − TS)\| | max \|KL − (β(F_data − F) − ln 6)\| |
|---|---|---|
| 50 × 500 | 4.8e-16 | 1.8e-15 |
| 10 × 5000 | 1.5e-15 | 7.1e-15 |

## Results

### 1. Individual F trajectories (500 epochs)

All 50 seeds form a single band around a slowly **rising** mean (−5.64 → −5.30).
Seeds 40 and 50 are ordinary members of that band. No sub-group of seeds follows
a visibly different path.

Context for every 500-epoch result: over this window the model is still
essentially the uniform distribution. `KL` stays at 0.98 ≈ ln(16/6) = 0.981,
which is the value of the uniform model, and `data_mass` stays at 0.375 = 6/16.
S stays within 2.7 % of ln 256. So the excursions in F are **not** the model
finding and losing good solutions. States with lower F do not have lower KL
(Spearman ρ(F, KL) = −0.16).

### 2. F_min vs F_final

`F_min ≤ F_final` holds for every run **by construction**, because the minimum
is taken over a set that includes the final epoch. Every point therefore lies
on or above the diagonal whatever the dynamics, and the grey half of panel (a)
is unreachable. "Most points lie above the diagonal" is not evidence for
anything. What carries information is the size of the gap and when the minimum
happens.

The test uses a noise-only null: the ensemble-mean trend plus a stationary
AR(1) process, fitted to each seed's own residual (lag-1 autocorrelation φ is
0.82–0.96, median 0.93, which gives a correlation time of about 14 epochs).
2000 surrogates were drawn per seed.

| | observed | null |
|---|---|---|
| gap `F_final − F_min`, median | 0.630 | 0.554 |
| epoch of `F_min`, median | 70 | 71 |
| seeds with gap above their own null 95th percentile | 7 / 50 | ≈ 2.5 expected |
| seed 40 gap (null 90 % range) | 0.563 (0.27–0.86), 52nd percentile | |
| seed 50 gap (null 90 % range) | 0.575 (0.29–0.85), 52nd percentile | |

The timing of the minima matches the null exactly, and seeds 40 and 50 are at
the median of their own nulls. The gaps are **modestly** larger than the null
predicts: a KS test of the percentiles gives D = 0.21, p = 0.026. This means the
fluctuations are somewhat burstier than a Gaussian AR(1) process. That excess is
the only departure from "noise around a common trend" in this data set.

### 3. Phase-space trajectories

* **500 epochs.** Every run starts near `(S, F) ≈ (ln 256, −5.6)` and diffuses
  into one shared, crescent-shaped cloud. There are no separate lobes or
  clusters, and seeds 40, 50 and the median seed (21) cover the same region. In
  the F–E plane the paths collapse onto a line of slope ≈ 1. That is expected:
  `F = E − TS` holds exactly and S barely changes, so the F–E plot is almost a
  relabelled E axis.
* **5000 epochs.** All 10 runs follow the same channel from the high-entropy
  plateau towards low S, E and F. They differ in how far along the channel they
  are, not in which channel they take.

Because `F = E − TS` is exact, (E, S) has only two independent coordinates, and
the F–S and F–E plots are two shears of the same (E, S) trajectory.

### 4. Distributions across seeds

| epoch | mean F | std | bimodality coeff. | Shapiro p |
|---|---|---|---|---|
| 50 | −5.565 | 0.159 | 0.42 | 0.34 |
| 100 | −5.512 | 0.157 | 0.35 | 0.23 |
| 250 | −5.430 | 0.150 | 0.47 | 0.10 |
| 500 | −5.303 | 0.150 | 0.40 | 0.37 |

Every distribution is unimodal (bimodality coefficient < 0.555) and consistent
with a Gaussian. It shifts upward at a constant width of about 0.15. E and S
behave the same way, except that S drops and spreads out at epoch 500 as the plateau starts
to end. None of this points to distinct classes of seeds.

### 5. Long runs: does it settle?

**Not by epoch 5000.** Learning starts around epoch 1200. By 5000, F has gone
from −5.62 to −17.08 ± 1.22 and KL from 0.985 to 0.175 ± 0.042, and every seed
is still drifting. Over the last 1000 epochs the median drift in F is −2.0,
against a residual scatter of 0.12, and all 10 of 10 seeds exceed 3σ. The same
seeds in `results/extended_10000ep` keep going: F = −19.8 at 7500 and −21.3 at
10 000, which is decelerating but not stationary.

This is the expected behaviour for this set-up. There is no weight decay, and
the maximum-likelihood solution for BAS pushes the model mass onto the 6
patterns, so |W| grows without bound and F (set by the lowest energies) goes to
−∞. A stationary thermodynamic state would need a regulariser (weight decay,
the `k=` argument in the old `Optimizer` calls) or a finite-capacity target.

### 6. Thermodynamics vs learning metrics

The exact identity checked above,

    D_KL(p_data ‖ p_model) = β ( ⟨F(v)⟩_data − F ) − ln 6,

says that **learning does not minimise the model free energy F. F appears in the
objective with a minus sign.** CD pushes the data free energy `⟨F(v)⟩_data` down
faster than F. The fall in F during learning is a side effect of the growing
weights, not the quantity being optimised. This supports the part of the
hypothesis that says learning is "not simply minimising the physical free
energy". It follows algebraically, so it can be stated as a derivation rather
than as an empirical finding.

Spearman correlations with KL (`tables/correlations.csv`):

| sweep | quantity | pooled levels | within-seed median | epoch-to-epoch increments |
|---|---|---|---|---|
| 500 | F | −0.16 | −0.15 | −0.13 |
| 500 | S | −0.10 | −0.12 | **−0.92** |
| 5000 | F | +0.95 | +0.97 | **−0.34** |
| 5000 | S | +0.98 | +0.98 | **−0.73** |
| 5000 | E | +0.97 | +0.98 | −0.46 |

Over the course of learning (5000-epoch levels), E, S and F all fall together
with KL. That is partly trivial, because they are all monotone in time. The
per-epoch *increments* have the **opposite** sign: a single CD step that lowers
S or F tends to *raise* KL. So on short timescales, the noise-driven
fluctuations move the model away from the data, and only the slow drift is
aligned with learning. That sign reversal between timescales is a concrete,
testable statement and may be worth building on.

## Caveats

* The null in §2 is a first-order model: a Gaussian AR(1) process around the
  ensemble mean. The small excess in gap size could come from non-Gaussian
  (bursty) increments rather than from any structure in the landscape.
* Everything here uses full-batch CD-1, so 500 epochs means 500 updates.
  `data_collection_a1.py` uses batch size 1 (6 updates per epoch) and 20 000
  epochs.
* `optimizer.py` is still the reconstructed plain-SGD version (see
  `results/README.md`).

## Reproducing

```bash
python results-2/run_thermo_metrics.py --epochs 500  --outdir results-2/runs_500ep                           # ~85 s
python results-2/run_thermo_metrics.py --epochs 5000 --seeds 1 2 3 4 5 6 7 8 9 10 --outdir results-2/runs_5000ep  # ~160 s
python results-2/plot_results2.py
```
