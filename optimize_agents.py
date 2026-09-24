"""
optimize_agents.py — байесовский подбор параметров трансляторов (gp_minimize).

Максимизируем суммарный PnL агентов по 4 параметрам трансляторов:
    T1: h_m1, h_r1;  T2: h_m2, h_r2.
Арбитражёры зафиксированы почти идеальными (FIXED_ARBS: h_mx = h_my = 1e-3,
шейдинг у них выключен в SimConfig): их параметры на оптимум не влияют, а при
шейдинге создавали артефакт марок (mean_field.md, разделы 8, 11, 12).
gp_minimize минимизирует, поэтому целевая функция возвращает -PnL.

Учёт — из SimConfig: PnL по справедливым маркам (X1 = X2 = 1, Y по мидам бирж),
взрыв = |PnL| агента выше BLOWUP_LIMIT (банкротство). Целевое значение точки —
СРЕДНЕЕ total по N_SEEDS прогонам, взорвавшийся прогон входит как
-BLOWUP_PENALTY (STAT = "mean"); медиана (STAT = "median") терпит до половины
лотерейных сидов и потому не рекомендуется.

Два режима (переключатель PARALLEL): последовательный gp_minimize либо
skopt.Optimizer с батчами ask(n_points) (constant liar) и счётом через joblib.

Границы поиска уже, чем раньше: по кривой flow_scan оптимум трансляторов при
order_size = 0.05 лежит около h_m ~ 1e-2, а при h_m < 1e-3 прогоны банкротятся
(глубина тонкой биржи). Шкала логарифмическая.

Запуск:  python optimize_agents.py [random_state]
"""

from __future__ import annotations

import csv
import os
import sys
import time

import numpy as np
from joblib import Parallel, delayed
from skopt import Optimizer, gp_minimize
from skopt.space import Real

from opt_simulation import PARAM_NAMES, opt_config, run_pnl

# --------------------------------------------------------------------------- #
# Настройки оптимизации
# --------------------------------------------------------------------------- #

OPT_NAMES = ("h_m1", "h_r1", "h_m2", "h_r2")
FIXED_ARBS = dict(h_mx=1e-3, h_my=1e-3)

H_M_BOUNDS = (0.0001, 10.0)
H_R_BOUNDS = (0.01, 100.0)

SPACE = [
    Real(*H_M_BOUNDS, prior="log-uniform", name="h_m1"),
    Real(*H_R_BOUNDS, prior="log-uniform", name="h_r1"),
    Real(*H_M_BOUNDS, prior="log-uniform", name="h_m2"),
    Real(*H_R_BOUNDS, prior="log-uniform", name="h_r2"),
]

N_CALLS = 200          # всего точек
N_INITIAL = 50        # из них случайных (разведка до включения GP)
RANDOM_STATE = 1      # воспроизводимость оптимизатора (можно задать аргументом)

PARALLEL = True
N_WORKERS = 12        # процессов = размер батча

TOTAL_STEPS = 3000
SEED: int | None = None   # seed рынка: None — новые случайные сиды на каждую точку
CRN_SEEDS: list[int] | None = None   # либо фиксированный список сидов для всех точек
N_SEEDS = 8               # прогонов на точку
STAT = "mean"             # "mean" | "median"

BLOWUP_LIMIT = 1000.0
BLOWUP_PENALTY = 1000.0

LOG_CSV = "opt_log_4d.csv"

AGENT_ORDER = ("T1", "T2", "AX", "AY")


def full_theta(x) -> list[float]:
    """4 параметра трансляторов -> вектор из 6 для run_pnl."""
    d = dict(zip(OPT_NAMES, (float(v) for v in x)))
    d.update(FIXED_ARBS)
    return [d[n] for n in PARAM_NAMES]


# --------------------------------------------------------------------------- #
# Целевая функция и печать
# --------------------------------------------------------------------------- #

def _seeds() -> list[int]:
    if CRN_SEEDS is not None:
        return list(CRN_SEEDS)
    if SEED is not None:
        return [SEED]
    return [int.from_bytes(os.urandom(4), "little") for _ in range(N_SEEDS)]


def evaluate(x, seeds: list[int] | None = None, n_jobs: int = 1) -> dict:
    """Оценка точки по сидам; целевое значение — STAT total с штрафом за взрыв."""
    seeds = seeds if seeds is not None else _seeds()
    theta = full_theta(x)
    t0 = time.perf_counter()
    if n_jobs > 1:
        runs = Parallel(n_jobs=min(n_jobs, len(seeds)))(
            delayed(run_pnl)(theta, cfg=opt_config(total_steps=TOTAL_STEPS, seed=s),
                             blowup_limit=BLOWUP_LIMIT) for s in seeds)
    else:
        runs = [run_pnl(theta, cfg=opt_config(total_steps=TOTAL_STEPS, seed=s),
                        blowup_limit=BLOWUP_LIMIT) for s in seeds]
    return summarize(runs, seeds, time.perf_counter() - t0)


def summarize(runs, seeds, elapsed: float = 0.0) -> dict:
    totals = [-BLOWUP_PENALTY if r["blown"] else r["total"] for r in runs]
    ok = [r for r in runs if not r["blown"]]
    if STAT == "mean":
        obj = float(np.mean(totals))
    elif STAT == "median":
        obj = float(np.median(totals))
    else:
        raise ValueError(f"STAT: {STAT!r}")
    return {
        "pnl": {k: (float(np.mean([r["pnl"][k] for r in ok])) if ok else 0.0)
                for k in AGENT_ORDER},
        "objective": obj,
        "mean": float(np.mean(totals)),
        "median": float(np.median(totals)),
        "std": float(np.std(totals)),
        "n_blown": len(runs) - len(ok),
        "seeds": list(seeds),
        "elapsed": elapsed,
    }


def _fmt_params(x) -> str:
    return "  ".join(f"{n}={v:.4g}" for n, v in zip(OPT_NAMES, x))


def _fmt_pnl(res: dict) -> str:
    parts = "  ".join(f"{k}={res['pnl'][k]:+8.4f}" for k in AGENT_ORDER)
    return f"{parts}  | {STAT}={res['objective']:+9.4f}"


class Log:
    """Печать каждой точки + строка в CSV + лучший результат."""

    def __init__(self, path: str, n_calls: int = N_CALLS):
        self.i = 0
        self.n_calls = n_calls
        self.best_obj = -np.inf
        self.best_x = None
        new = not os.path.exists(path)
        self._f = open(path, "a", newline="", encoding="utf-8")
        self._csv = csv.writer(self._f)
        if new:
            self._csv.writerow(("iter",) + PARAM_NAMES + AGENT_ORDER
                               + ("objective", "mean", "median", "std", "n_blown",
                                  "seeds", "elapsed_s"))

    def record(self, x, res: dict) -> None:
        self.i += 1
        star = ""
        if res["objective"] > self.best_obj:
            self.best_obj, self.best_x = res["objective"], list(x)
            star = " <-- новый лучший"
        print(f"[{self.i:>3}/{self.n_calls}] {_fmt_params(x)}")
        blown_txt = (f"  ВЗОРВАЛОСЬ {res['n_blown']}/{len(res['seeds'])}"
                     if res["n_blown"] else "")
        print(f"        {_fmt_pnl(res)}  mean={res['mean']:+.4f} ±{res['std']:.4f}"
              f"{blown_txt}  (seeds={','.join(map(str, res['seeds']))}, "
              f"{res['elapsed']:.1f}s){star}")
        self._csv.writerow([self.i] + [f"{v:.6g}" for v in full_theta(x)]
                           + [f"{res['pnl'][k]:.6f}" for k in AGENT_ORDER]
                           + [f"{res['objective']:.6f}", f"{res['mean']:.6f}",
                              f"{res['median']:.6f}", f"{res['std']:.6f}",
                              res["n_blown"], ";".join(map(str, res["seeds"])),
                              f"{res['elapsed']:.2f}"])
        self._f.flush()

    def close(self) -> None:
        self._f.close()


# --------------------------------------------------------------------------- #
# Режимы оптимизации
# --------------------------------------------------------------------------- #

def run_sequential(log: Log, random_state: int = RANDOM_STATE):
    def objective(x):
        res = evaluate(x, n_jobs=N_WORKERS)
        log.record(x, res)
        return -res["objective"]
    return gp_minimize(objective, SPACE, n_calls=N_CALLS, n_initial_points=N_INITIAL,
                       acq_func="EI", random_state=random_state, verbose=False)


def run_parallel(log: Log, random_state: int = RANDOM_STATE):
    """ask/tell-цикл: батч точек x сиды считаются одной параллельной пачкой."""
    opt = Optimizer(SPACE, base_estimator="GP", n_initial_points=N_INITIAL,
                    acq_func="EI", random_state=random_state)
    done = 0
    while done < N_CALLS:
        batch = min(N_WORKERS, N_CALLS - done)
        xs = opt.ask(n_points=batch, strategy="cl_min")
        seed_sets = [_seeds() for _ in xs]
        jobs = [(k, s) for k, seeds in enumerate(seed_sets) for s in seeds]
        t0 = time.perf_counter()
        runs = Parallel(n_jobs=N_WORKERS, verbose=10)(
            delayed(run_pnl)(full_theta(xs[k]),
                             cfg=opt_config(total_steps=TOTAL_STEPS, seed=s),
                             blowup_limit=BLOWUP_LIMIT) for k, s in jobs)
        el = time.perf_counter() - t0
        results = []
        for k, seeds in enumerate(seed_sets):
            rr = [r for (kk, _), r in zip(jobs, runs) if kk == k]
            results.append(summarize(rr, seeds, el / len(xs)))
        opt.tell(xs, [-r["objective"] for r in results])
        for x, r in zip(xs, results):
            log.record(x, r)
        done += batch
        print(f"    лучший на данный момент: {STAT}={log.best_obj:+.4f}  "
              f"({_fmt_params(log.best_x)})")
    return opt.get_result()


# --------------------------------------------------------------------------- #

def main() -> None:
    random_state = int(sys.argv[1]) if len(sys.argv) > 1 else RANDOM_STATE
    mode = f"параллельный, {N_WORKERS} процессов" if PARALLEL else "последовательный"
    print("=" * 78)
    print(f"gp_minimize: {N_CALLS} точек ({N_INITIAL} случайных), режим: {mode}, "
          f"random_state={random_state}")
    seed_txt = (f"CRN {CRN_SEEDS}" if CRN_SEEDS is not None else
                (f"случайные, {N_SEEDS} на точку" if SEED is None else f"{SEED} (фиксированный)"))
    print(f"прогон: {TOTAL_STEPS} тиков, сиды: {seed_txt}; критерий: {STAT}; лог: {LOG_CSV}")
    print(f"границы: h_m in {H_M_BOUNDS}, h_R in {H_R_BOUNDS} (лог-шкала); "
          f"арбитражёры фиксированы: {FIXED_ARBS}")
    print("=" * 78)

    log = Log(LOG_CSV)
    t0 = time.perf_counter()
    try:
        result = run_parallel(log, random_state) if PARALLEL else run_sequential(log, random_state)
    finally:
        log.close()
    elapsed = time.perf_counter() - t0

    best_x, best_obj = result.x, -result.fun
    print()
    print("=" * 78)
    print(f"Готово за {elapsed / 60:.1f} мин ({log.i} точек)")
    print(f"Лучший суммарный PnL ({STAT} по сидам): {best_obj:+.4f}")
    for name, value in zip(OPT_NAMES, best_x):
        print(f"    {name} = {value:.4g}")
    res = evaluate(best_x, seeds=list(range(901, 913)), n_jobs=N_WORKERS)
    print(f"Контрольный прогон (12 свежих сидов): {_fmt_pnl(res)}  mean={res['mean']:+.3f} "
          f"±{res['std']:.3f}  взрывов {res['n_blown']}/12")
    print("=" * 78)


if __name__ == "__main__":
    main()
