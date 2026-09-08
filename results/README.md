# RBM learning as thermodynamic relaxation

Data, plotting code and preliminary figures for the manuscript panel asking
whether RBM training can be read as a thermodynamic relaxation process.

## What was run

`run_thermo_runs.py` trains the RBM in `my_RBM_tf2_test.py` and, after **every**
epoch, evaluates the exact thermodynamics of the model's Gibbs distribution by
full enumeration of all `2^4 x 2^4 = 256` joint states, using the routines that
are already in the RBM class:

| quantity | how it is computed |
|---|---|
| `E_matrix` | `machine.energyr(v_all, h_all)[1]` — `E(v,h)` for all 256 states |
| `Z` | `machine.True_Z(v_all, h_all, Beta)` |
| `E` = ⟨E⟩ | `machine.expected_energy_2(E_matrix, Z, Beta)` |
| `S` | `machine.thermo_entropy(E_matrix, Z, Beta)[0]` (`k_B = 1`) |
| `F` | `-(1/Beta) * log(Z)` |
| `F_check` | `E - T*S` |
| `err` | `abs(F - (E - T*S))` |
| `delta_F` | `F[n+1] - F[n]` (NaN on the last epoch) |

`v_all` and `h_all` come from `datasets.bas_data.get_everything_2`, i.e. all 16
binary configurations of each layer.

### Configuration (primary run)

| setting | value |
|---|---|
| dataset | 2×2 Bars-and-Stripes, 6 unique patterns |
| `visible_dim` | 4 |
| `hidden_dim` | 4 |
| `initial_temperature` | 1.0 |
| `annealing_decay` | 0 → `T = 1` for every epoch |
| training algorithm | CD, `k = 1` |
| epochs | 500 |
| `batch_size` | `len(x_train)` = 6 → **one gradient update per epoch** |
| learning rate | 0.15 (the value used throughout `data_collection_a*.py`) |
| seeds | 1 … 50, each seeding `random`, `numpy` and `tensorflow` before the RBM is built |

## Files

```
results/
  run_thermo_runs.py            simulation driver
  plot_thermo_figure.py         plotting / analysis script
  all_runs.csv                  master file, 50 seeds x 500 epochs = 25 000 rows
  data/run_001.csv … run_050.csv  one file per seed
  epoch_summary.csv             per-epoch mean & std across the 50 seeds
  summary.txt                   consistency check + start/end values
  consistency_report.txt        written by the run script
  figures/
    fig_thermo_relaxation.png/.pdf   the three-panel manuscript figure
    energy_vs_epoch.png              panel (a) on its own
    entropy_vs_epoch.png             panel (b) on its own
    free_energy_vs_epoch.png         panel (c) on its own
    delta_F_diagnostic.png           is delta_F < 0 for most of training?
  extended_10000ep/             identical sweep run out to 10 000 epochs (see below)
```

CSV columns: `seed, epoch, temperature, Z, E, S, F, F_check, err, delta_F`.

## Consistency check

`F` and `E - T*S` agree to machine precision over all 25 000 rows:

```
mean |F - (E - T*S)| : 4.834533e-16
max  |F - (E - T*S)| : 2.664535e-15
mean relative error  : 8.896592e-17
```

This is expected — `thermo_entropy` and `expected_energy_2` are both built from
the same Boltzmann weights `exp(-beta E)/Z`, so `F = E - T S` is an algebraic
identity here, not an independent test of the model. It does confirm there is no
bug in the enumeration or in the `Beta` bookkeeping.

## Result of the primary (500-epoch) run

| quantity | epoch 1 | epoch 500 |
|---|---|---|
| ⟨E⟩ | −0.104 | +0.150 |
| S | 5.5377 | 5.4532 |
| F | −5.6416 ± 0.0842 | −5.3032 ± 0.1495 |

`delta_F < 0` on **49.2 %** of epoch steps, and the mean `delta_F` is *positive*
(`+6.8e-4`). The increments are a symmetric noise distribution centred just
above zero — there is no relaxation signal.

**The reason is that the model is essentially untrained after 500 epochs.**
With `batch_size = len(x_train)` there is exactly one gradient update per epoch,
so 500 epochs is 500 updates. The entropy is still 5.45 against a
maximum-entropy value of `ln(256) = 5.5452`, i.e. the RBM is still within 2 % of
the uniform distribution, and the exact log-likelihood per BAS pattern is
−2.771, indistinguishable from the uniform-over-16 value `ln(1/16) = −2.7726`.
The 500-epoch window sits entirely inside the pre-learning plateau, and what the
figure shows is CD-1 sampling noise plus a slow drift, not relaxation. For
comparison, `data_collection_a1.py` uses `batch_size = 1` with 20 000 epochs,
i.e. ≈120 000 updates.

## Supplementary run: 10 000 epochs

`results/extended_10000ep/` repeats the identical sweep (same 50 seeds, same
hyper-parameters) with `--epochs 10000`, which is long enough for learning to
actually happen. This is the run that answers the question.

| quantity | epoch 1 | epoch 10 000 |
|---|---|---|
| ⟨E⟩ | −0.104 | −18.875 |
| S | 5.5377 | 2.0553 |
| F | −5.6416 ± 0.0842 | −20.930 ± 2.377 |

All three panels are monotone decreasing after a plateau, with a relaxation
onset around epoch ≈1 500 and saturation by ≈8 000. Consistency error is
3.94e-15 over 500 000 rows.

### How to state the `delta_F < 0` claim

The naive statistic is misleading. Per single epoch, `delta_F < 0` on only
**30.2 %** of steps — but the *mean* `delta_F` is negative (−1.53e-3) and
**every one of the 50 seeds** ends with `F(last) < F(first)`. The reason is that
the single-epoch increment is dominated by CD-1 sampling noise (median exactly
0, because a converged full-batch CD-1 update is frequently the zero vector),
while the drift is small and asymmetric — a few large negative steps against
many tiny positive ones. Coarse-graining recovers the monotone picture:

| window `W` (epochs) | fraction with `F[n+W] − F[n] < 0` |
|---|---|
| 1 | 0.302 |
| 5 | 0.508 |
| 10 | 0.584 |
| 50 | 0.729 |
| 100 | 0.801 |
| 500 | 0.930 |
| 1 000 | 0.975 |
| whole run | **1.000** (50/50 seeds) |

So the defensible statement for the manuscript is *"the free energy decreases
monotonically on the timescale of learning (93 % of 500-epoch windows, 100 % of
runs end-to-end); the per-epoch increment is noise-dominated"* — not *"delta_F <
0 for most epochs"*. `figures/delta_F_diagnostic.png` panel (c) is that table as
a plot.

For contrast, in the 500-epoch run the same coarse-graining goes the *other*
way (`W = 200` → 0.32, and only 1 of 50 seeds ends below where it started),
which is the signature of a run that never left the plateau.

## Reproducing

```bash
python results/run_thermo_runs.py                       # 50 seeds x 500 epochs (~90 s)
python results/plot_thermo_figure.py                    # figures + summary

# supplementary
python results/run_thermo_runs.py --epochs 10000 --outdir results/extended_10000ep
python results/plot_thermo_figure.py --indir results/extended_10000ep
```

## Two repo files that had to be added

`my_RBM_tf2_test.py` and `data_collection_a*.py` import modules that were not
part of the upload, so they were reconstructed at the repo root:

* **`optimizer.py`** — the `Optimizer(machine, learning_rate)` / `optimizer.fit()`
  interface that `RBM.train` calls. It is plain SGD *ascent*
  (`theta <- theta + lr * grad`), which is the correct sign for the CD gradients
  returned by `parallel_cd` / `contr_divergence`. **If the original
  `optimizer.py` had momentum, weight decay or a learning-rate schedule, these
  numbers will differ and the run should be repeated with it.**
* **`datasets/bas_data.py`** — a thin re-export of the root-level `bas_data.py`,
  so that `from datasets.bas_data import ...` resolves.

`run_thermo_runs.py` also stubs out `tf.summary.create_file_writer`, because
`RBM.__init__` unconditionally opens a TensorBoard writer under a hard-coded
`D:\Desktop\...` path. It runs its own epoch loop, which mirrors the inner loop
of `RBM.train` (shuffle → `parallel_cd` → `optimizer.fit()`) minus the
per-epoch `save_model()` call, so that the thermodynamics can be sampled between
epochs.
