"""
Thermodynamics of RBM learning -- data generation.

Trains a small RBM on the 2x2 Bars-and-Stripes (BAS) dataset with Jack's code
(``my_RBM_tf2_test.RBM``) and records the exact thermodynamic quantities of the
RBM Gibbs distribution after every epoch.

Configuration (as specified for the manuscript figure)
------------------------------------------------------
    dataset             2x2 BAS  (6 unique patterns)
    visible_dim         4
    hidden_dim          4
    initial_temperature 1.0
    annealing_decay     0        (no annealing -> T = 1 for all epochs)
    CD steps k          1
    epochs              500
    batch_size          len(x_train)   (full batch -> 1 update / epoch)
    learning_rate       0.15     (the value used throughout data_collection_a*.py)
    seeds               1 .. 50

Quantities recorded each epoch, using the routines already in the RBM class:
    E_matrix        = machine.energyr(v_all, h_all)[1]      # E(v,h) over all 16x16 states
    Z               = machine.True_Z(v_all, h_all, Beta)
    E   = <E>       = machine.expected_energy_2(E_matrix, Z, Beta)
    S               = machine.thermo_entropy(E_matrix, Z, Beta)[0]   # k_B = 1
    F               = -(1/Beta) * log(Z)
    F_check         = E - T*S
    err             = |F - (E - T*S)|
    delta_F[n]      = F[n+1] - F[n]   (NaN on the last epoch)

Outputs
-------
    results/data/run_001.csv ... run_050.csv    one file per seed
    results/all_runs.csv                        master file (all seeds)
    results/consistency_report.txt              F vs E - T*S check

Usage
-----
    python results/run_thermo_runs.py                 # full 50-seed production run
    python results/run_thermo_runs.py --seeds 1 2 --epochs 20 --tag smoke
"""

import argparse
import os
import random
import sys
import time

import numpy as np

# --- repo root on the path so that `my_RBM_tf2_test` / `optimizer` import ------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf  # noqa: E402


# RBM.__init__ unconditionally opens a TensorBoard writer under a hard-coded
# 'D:\\Desktop\\...' path.  The summaries are unused here, so stub the writer out
# before the machine is instantiated.
class _NullWriter:
    def set_as_default(self):
        return None

    def as_default(self, *a, **kw):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def close(self):
        return None


tf.summary.create_file_writer = lambda *a, **kw: _NullWriter()

import pandas as pd  # noqa: E402
from datasets.bas_data import get_data, get_everything_2  # noqa: E402
from my_RBM_tf2_test import RBM  # noqa: E402
from optimizer import Optimizer  # noqa: E402


COLUMNS = ["seed", "epoch", "temperature", "Z", "E", "S", "F", "F_check", "err", "delta_F"]


def run_one_seed(seed, epochs, learning_rate, k, visible_dim, hidden_dim,
                 initial_temperature, annealing_decay):
    """Train one RBM and return a DataFrame with one row per epoch."""
    # ---- seed everything *before* the RBM is created -------------------------
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # 2x2 BAS.  get_data() returns the unique BAS patterns, so train and test are
    # the same 6 states -- this is the fully enumerable toy case.
    x_train = get_data(np.random, s=2)
    x_test = get_data(np.random, s=2)
    np.random.shuffle(x_train)
    np.random.shuffle(x_test)
    data = {"x_train": x_train, "x_test": x_test}

    batch_size = len(x_train)

    machine = RBM(visible_dim, hidden_dim, epochs, (2, 2), batch_size,
                  k=k, n_test_samples=len(x_test), small_Big=True,
                  NAME="BAS_2x2_thermo_seed%03d" % seed, l_1=0,
                  non_parallel=False,
                  initial_temperature=initial_temperature,
                  annealing_decay=annealing_decay)
    optimus = Optimizer(machine, learning_rate)

    # all 2^4 configurations of each layer -> exact partition function
    v_all = np.asarray(get_everything_2(visible_dim), dtype=np.float64)
    h_all = np.asarray(get_everything_2(hidden_dim), dtype=np.float64)

    rows = []
    for epoch in range(1, epochs + 1):
        machine.epoch = epoch  # drives sigmoid_Temp / the annealing schedule

        # --- one training epoch: mirrors the inner loop of RBM.train ----------
        np.random.shuffle(data["x_train"])
        for i in range(0, data["x_train"].shape[0], machine._batch_size):
            x_mini = data["x_train"][i:i + machine._batch_size]
            dw, dvb, dhb = machine.parallel_cd(x_mini)
            machine.grad_dict = {"weights": dw,
                                 "visible_biases": dvb,
                                 "hidden_biases": dhb}
            optimus.fit()

        # --- thermodynamics of the current model -----------------------------
        T = machine.initial_temperature * np.exp(-machine.annealing_decay * machine.epoch)
        Beta = 1.0 / (T * machine.boltzmann)

        _, E_matrix = machine.energyr(v_all, h_all)
        Z = float(machine.True_Z(v_all, h_all, Beta))
        E = float(machine.expected_energy_2(E_matrix, Z, Beta))
        S = float(machine.thermo_entropy(E_matrix, Z, Beta)[0])
        F = float(-(1.0 / Beta) * np.log(Z))
        F_check = E - T * S

        rows.append((seed, epoch, T, Z, E, S, F, F_check, abs(F - F_check), np.nan))

    df = pd.DataFrame(rows, columns=COLUMNS)
    # delta_F[n] = F[n+1] - F[n]; undefined on the final epoch
    df["delta_F"] = df["F"].shift(-1) - df["F"]
    return df


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 51)))
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--learning-rate", type=float, default=0.15)
    p.add_argument("--k", type=int, default=1)
    p.add_argument("--visible-dim", type=int, default=4)
    p.add_argument("--hidden-dim", type=int, default=4)
    p.add_argument("--initial-temperature", type=float, default=1.0)
    p.add_argument("--annealing-decay", type=float, default=0.0)
    p.add_argument("--outdir", default=os.path.join(ROOT, "results"))
    p.add_argument("--tag", default="", help="suffix for the master csv / report")
    args = p.parse_args()

    datadir = os.path.join(args.outdir, "data")
    os.makedirs(datadir, exist_ok=True)

    all_frames = []
    t0 = time.time()
    for seed in args.seeds:
        t1 = time.time()
        df = run_one_seed(seed, args.epochs, args.learning_rate, args.k,
                          args.visible_dim, args.hidden_dim,
                          args.initial_temperature, args.annealing_decay)
        df.to_csv(os.path.join(datadir, "run_%03d.csv" % seed), index=False)
        all_frames.append(df)
        print("seed %3d done in %6.1f s | mean |F-(E-TS)| = %.3e | F: %+.4f -> %+.4f"
              % (seed, time.time() - t1, df["err"].mean(),
                 df["F"].iloc[0], df["F"].iloc[-1]), flush=True)

    master = pd.concat(all_frames, ignore_index=True)
    tag = ("_" + args.tag) if args.tag else ""
    master_path = os.path.join(args.outdir, "all_runs%s.csv" % tag)
    master.to_csv(master_path, index=False)

    # ---- consistency check ---------------------------------------------------
    frac_neg = float((master["delta_F"].dropna() < 0).mean())
    report = [
        "Consistency check: F  vs  E - T*S",
        "=" * 46,
        "runs                 : %d seeds x %d epochs = %d rows"
        % (len(args.seeds), args.epochs, len(master)),
        "mean |F - (E - T*S)| : %.6e" % master["err"].mean(),
        "max  |F - (E - T*S)| : %.6e" % master["err"].max(),
        "median               : %.6e" % master["err"].median(),
        "mean relative error  : %.6e" % (master["err"] / master["F"].abs()).mean(),
        "",
        "fraction of epochs with delta_F < 0 : %.4f" % frac_neg,
        "master file          : %s" % master_path,
    ]
    txt = "\n".join(report)
    print("\n" + txt)
    with open(os.path.join(args.outdir, "consistency_report%s.txt" % tag), "w") as fh:
        fh.write(txt + "\n")
    print("\ntotal wall time: %.1f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
