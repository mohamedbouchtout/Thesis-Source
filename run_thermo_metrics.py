"""
Thermodynamics + learning metrics of 3x3 RBM training.

Same model, data, seeding and training loop as ``results/run_thermo_runs.py``
(3x3 BAS, 9 visible / 9 hidden, T = 1, no annealing, CD-1, full batch,
lr = 0.15), but every epoch it additionally records learning metrics.

Everything is computed *exactly* by enumerating the 512 x 512 joint states, with
pure numpy -- no random numbers are drawn outside the training step, so the
training trajectory for a given seed is bit-identical to the results/ run.

Per-epoch columns
-----------------
    seed, epoch, temperature
    Z, E, S, F, F_check, err, delta_F      as in results/ (F = -T ln Z, F_check = E - T S)
    F_data     mean visible free energy of the training patterns,
               F(v) = -T ln sum_h exp(-E(v,h)/T), averaged over the 6 BAS patterns
    KL         D_KL(p_data || p_model) = sum_d (1/14) ln[(1/14) / p_model(d)]
               (same definition as RBM.KL_div_BIG_SMALL; checked against it)
    NLL        -(1/14) sum_d ln p_model(d)   (= KL + ln 14)
    data_mass  sum_d p_model(d): model probability mass on the 6 BAS patterns
    recon_sq   exact expectation of RBM.average_squared_error over v->h->v'
               (mean over patterns and units of (v - v')^2)
    recon_ce   exact expectation of RBM.recon_c_e
               (binary cross-entropy of v against p(v'|h), h ~ p(h|v), summed
               over units, averaged over patterns)

Identity used as a check: KL = beta * (F_data - F) - ln 14.

Usage
-----
    python run_thermo_metrics.py --epochs 500 --outdir <dir>/runs_500ep
    python run_thermo_metrics.py --epochs 5000 --seeds 1 2 3 4 5 6 7 8 9 10 --outdir <dir>/runs_5000ep
"""

import argparse
import os
import random
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf  # noqa: E402


# RBM.__init__ opens a TensorBoard writer under a hard-coded D:\ path; stub it.
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


COLUMNS = ["seed", "epoch", "temperature", "Z", "E", "S", "F", "F_check", "err",
           "delta_F", "F_data", "KL", "NLL", "data_mass", "recon_sq", "recon_ce"]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _logsumexp(a, axis):
    m = np.max(a, axis=axis, keepdims=True)
    return np.squeeze(m, axis) + np.log(np.sum(np.exp(a - m), axis=axis))


def learning_metrics(W, b, c, E_matrix, Beta, h_all, data_idx, x_data):
    """Exact KL / NLL / reconstruction metrics. Pure numpy, no RNG.

    W : (h, v)   b : (v,)   c : (h,)   E_matrix : (16 v-states, 16 h-states)
    """
    T = 1.0 / Beta
    logw = -Beta * E_matrix                       # (nv, nh)
    logZ = _logsumexp(logw.ravel(), axis=0)
    log_pv = _logsumexp(logw, axis=1) - logZ      # ln p_model(v) for all 16 v
    Fv = -T * _logsumexp(logw, axis=1)            # visible free energy F(v)

    n = len(data_idx)
    lp_d = log_pv[data_idx]
    KL = float(np.mean(np.log(1.0 / n) - lp_d))
    NLL = float(-np.mean(lp_d))
    data_mass = float(np.exp(lp_d).sum())
    F_data = float(np.mean(Fv[data_idx]))

    # reconstruction v -> h -> v', exact expectation over h ~ p(h|v)
    ph = _sigmoid(Beta * (x_data @ W.T + c))                      # (n, h)
    p_h_given_v = np.prod(np.where(h_all[None, :, :] == 1,
                                   ph[:, None, :], 1.0 - ph[:, None, :]), axis=2)  # (n, nh)
    q = _sigmoid(Beta * (h_all @ W + b))                          # (nh, v) = p(v'_i = 1 | h)
    xv = x_data[:, None, :]                                       # (n, 1, v)
    sq = xv * (1.0 - q[None]) + (1.0 - xv) * q[None]              # E[(v - v')^2 | h] per unit
    recon_sq = float(np.mean(np.sum(p_h_given_v[:, :, None] * sq, axis=1)))
    eps = 1e-12
    ce = -(xv * np.log(np.clip(q[None], eps, 1.0))
           + (1.0 - xv) * np.log(np.clip(1.0 - q[None], eps, 1.0)))
    recon_ce = float(np.mean(np.sum(p_h_given_v * ce.sum(axis=2), axis=1)))
    return F_data, KL, NLL, data_mass, recon_sq, recon_ce


def run_one_seed(seed, epochs, learning_rate, k, visible_dim, hidden_dim,
                 initial_temperature, annealing_decay):
    # ---- seed everything *before* the RBM is created (identical to results/) --
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    dim = int(visible_dim ** 0.5)
    x_train = get_data(np.random, s=dim)
    x_test = get_data(np.random, s=dim)
    np.random.shuffle(x_train)
    np.random.shuffle(x_test)
    data = {"x_train": x_train, "x_test": x_test}
    batch_size = len(x_train)

    machine = RBM(visible_dim, hidden_dim, epochs, (dim, dim), batch_size,
                  k=k, n_test_samples=len(x_test), small_Big=True,
                  NAME=f"BAS_{dim}x{dim}_thermo_seed%03d" % seed, l_1=0,
                  non_parallel=False,
                  initial_temperature=initial_temperature,
                  annealing_decay=annealing_decay)
    optimus = Optimizer(machine, learning_rate)

    v_all = np.asarray(get_everything_2(visible_dim), dtype=np.float64)
    h_all = np.asarray(get_everything_2(hidden_dim), dtype=np.float64)

    # The unique 3x3 BAS patterns and their row index in v_all.
    x_data = np.unique(np.asarray(x_train, dtype=np.float64), axis=0)
    data_idx = np.array([int(np.flatnonzero((v_all == d).all(1))[0]) for d in x_data])

    rows = []
    for epoch in range(1, epochs + 1):
        machine.epoch = epoch

        # --- one training epoch: identical to results/run_thermo_runs.py -------
        np.random.shuffle(data["x_train"])
        for i in range(0, data["x_train"].shape[0], machine._batch_size):
            x_mini = data["x_train"][i:i + machine._batch_size]
            dw, dvb, dhb = machine.parallel_cd(x_mini)
            machine.grad_dict = {"weights": dw,
                                 "visible_biases": dvb,
                                 "hidden_biases": dhb}
            optimus.fit()

        # --- thermodynamics (same calls as results/) --------------------------
        T = machine.initial_temperature * np.exp(-machine.annealing_decay * machine.epoch)
        Beta = 1.0 / (T * machine.boltzmann)
        _, E_matrix = machine.energyr(v_all, h_all)
        Z = float(machine.True_Z(v_all, h_all, Beta))
        E = float(machine.expected_energy_2(E_matrix, Z, Beta))
        S = float(machine.thermo_entropy(E_matrix, Z, Beta)[0])
        F = float(-(1.0 / Beta) * np.log(Z))
        F_check = E - T * S

        # --- learning metrics (numpy only, no RNG) ---------------------------
        W = np.asarray(machine.weights, dtype=np.float64)
        b = np.asarray(machine.visible_biases, dtype=np.float64).ravel()
        c = np.asarray(machine.hidden_biases, dtype=np.float64).ravel()
        Em = np.asarray(E_matrix, dtype=np.float64)
        F_data, KL, NLL, data_mass, recon_sq, recon_ce = learning_metrics(
            W, b, c, Em, Beta, h_all, data_idx, x_data)

        if epoch == 1 and abs(T - 1.0) < 1e-12:
            # cross-check against the RBM class's own exact KL routine
            # (deterministic, hard-codes Beta = 1 so only valid at T = 1)
            kl_ref = float(np.asarray(machine.KL_div_BIG_SMALL(v_all, h_all, x_data)).ravel()[0])
            assert abs(kl_ref - KL) < 1e-10, (kl_ref, KL)

        rows.append((seed, epoch, T, Z, E, S, F, F_check, abs(F - F_check), np.nan,
                     F_data, KL, NLL, data_mass, recon_sq, recon_ce))

    df = pd.DataFrame(rows, columns=COLUMNS)
    df["delta_F"] = df["F"].shift(-1) - df["F"]
    return df


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 51)))
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--learning-rate", type=float, default=0.15)
    p.add_argument("--k", type=int, default=1)
    p.add_argument("--visible-dim", type=int, default=9)
    p.add_argument("--hidden-dim", type=int, default=9)
    p.add_argument("--initial-temperature", type=float, default=1.0)
    p.add_argument("--annealing-decay", type=float, default=0.0)
    p.add_argument("--outdir")
    args = p.parse_args()

    datadir = os.path.join(args.outdir, "data")
    os.makedirs(datadir, exist_ok=True)

    frames = []
    t0 = time.time()
    for seed in args.seeds:
        t1 = time.time()
        df = run_one_seed(seed, args.epochs, args.learning_rate, args.k,
                          args.visible_dim, args.hidden_dim,
                          args.initial_temperature, args.annealing_decay)
        df.to_csv(os.path.join(datadir, "run_%03d.csv" % seed), index=False)
        frames.append(df)
        print("seed %3d  %6.1f s | F %+.4f -> %+.4f | KL %.4f -> %.4f"
              % (seed, time.time() - t1, df.F.iloc[0], df.F.iloc[-1],
                 df.KL.iloc[0], df.KL.iloc[-1]), flush=True)

    master = pd.concat(frames, ignore_index=True)
    master.to_csv(os.path.join(args.outdir, "all_runs.csv"), index=False)

    beta = 1.0 / master["temperature"]
    n_data = 14  # 3x3 Bars-and-Stripes has 2^3 + 2^3 - 2 unique patterns.
    ident = (master["KL"] - (beta * (master["F_data"] - master["F"]) - np.log(n_data))).abs()
    report = [
        "Consistency checks  (%d seeds x %d epochs = %d rows)"
        % (len(args.seeds), args.epochs, len(master)),
        "mean |F - (E - T*S)|               : %.6e" % master["err"].mean(),
        "max  |F - (E - T*S)|               : %.6e" % master["err"].max(),
        "max  |KL - (beta(F_data-F) - ln14)| : %.6e" % ident.max(),
        "KL matched RBM.KL_div_BIG_SMALL at epoch 1 of every seed (|diff| < 1e-10)",
    ]
    txt = "\n".join(report)
    print("\n" + txt + "\nwall time %.1f s" % (time.time() - t0))
    with open(os.path.join(args.outdir, "consistency_report.txt"), "w") as fh:
        fh.write(txt + "\n")


if __name__ == "__main__":
    main()
