"""
optimize_agents.py — байесовский подбор параметров 4 агентов (gp_minimize).

Максимизируем суммарный PnL агентов T1, T2, AX, AY по 6 параметрам:
    T1: h_m1, h_r1;  T2: h_m2, h_r2;  AX: h_mx;  AY: h_my.
gp_minimize минимизирует, поэтому целевая функция возвращает -PnL.

Два режима (переключатель PARALLEL ниже):
    * последовательный — классический gp_minimize: каждая следующая точка
      выбирается с учётом всех предыдущих (максимальная эффективность GP);
    * параллельный — skopt.Optimizer: ask(n_points) выдаёт батч точек
      (стратегия constant liar "cl_min"), они считаются параллельно через
      joblib, затем tell. Быстрее по времени на стену, но GP получает
      информацию батчами, так что на то же число прогонов сходимость
      чуть хуже. Число процессов — N_WORKERS.

Шум: при SEED=None рынок каждый раз случайный, поэтому одна точка оценивается
N_SEEDS прогонами с разными сидами. Оптимизируется МЕДИАНА total по сидам —
в отличие от среднего она устойчива к тяжёлым хвостам: один сверхудачный
прогон не вытягивает точку наверх. Остаточный шум GP учитывает сам
(noise="gaussian" в суррогатной модели).

Защита от вырожденных режимов: при экстремальных параметрах (lam агентов
различаются на порядки) клиринг CE плохо обусловлен и цены/PnL идут вразнос —
PnL превращается в лотерею ±1e5. Такой прогон обрывается, как только |PnL|
агента превышает BLOWUP_LIMIT, и учитывается со штрафом -BLOWUP_PENALTY,
поэтому GP видит в этой области стабильно плохое значение, а не лотерею.

Каждая точка печатается: параметры, средний PnL каждого агента, сумма ± разброс,
лучший результат на текущий момент. Всё также пишется в CSV (LOG_CSV).

Запуск:  python optimize_agents.py
"""

from __future__ import annotations

import csv
import os
import time

import numpy as np
from joblib import Parallel, delayed
from skopt import Optimizer, gp_minimize
from skopt.space import Real

from opt_simulation import PARAM_NAMES, opt_config, run_pnl

# --------------------------------------------------------------------------- #
# Настройки оптимизации
# --------------------------------------------------------------------------- #

# границы поиска; шкала логарифмическая (prior="log-uniform"): стартовые
# точки сэмплируются равномерно по декадам, а GP работает в лог-координатах —
# иначе при границах в несколько порядков почти все точки падали бы в верхнюю
# декаду. Границы обязаны быть строго положительными.
H_M_BOUNDS = (0.0001, 300.0)
H_R_BOUNDS = (0.0001, 300.0)

SPACE = [
    Real(*H_M_BOUNDS, prior="log-uniform", name="h_m1"),
    Real(*H_R_BOUNDS, prior="log-uniform", name="h_r1"),
    Real(*H_M_BOUNDS, prior="log-uniform", name="h_m2"),
    Real(*H_R_BOUNDS, prior="log-uniform", name="h_r2"),
    Real(*H_M_BOUNDS, prior="log-uniform", name="h_mx"),
    Real(*H_M_BOUNDS, prior="log-uniform", name="h_my"),
]

N_CALLS = 100          # всего прогонов симуляции
N_INITIAL = 20        # из них случайных (разведка до включения GP)
RANDOM_STATE = 1      # воспроизводимость оптимизатора

PARALLEL = True       # False — чистый последовательный gp_minimize
N_WORKERS = 12         # процессов в параллельном режиме (= размер батча)

TOTAL_STEPS = 6000    # тиков в одном прогоне
SEED: int | None = None   # seed рынка: None — новый случайный на каждый прогон
                          # (целевая функция шумная, GP это учитывает через
                          # noise="gaussian"); число — фиксированный seed,
                          # целевая функция детерминирована
N_SEEDS = 5              # прогонов с разными сидами на одну точку; целевое
                          # значение — медиана total по ним; действует
                          # только при SEED=None

BLOWUP_LIMIT = 1000.0     # порог |PnL| агента (= стартовый капитал c0):
                          # выше — симуляция пошла вразнос, прогон обрывается
BLOWUP_PENALTY = 1000.0   # штрафной total взорвавшегося прогона (со знаком -)

LOG_CSV = "opt_log.csv"

AGENT_ORDER = ("T1", "T2", "AX", "AY")


# --------------------------------------------------------------------------- #
# Целевая функция и печать
# --------------------------------------------------------------------------- #

def evaluate(x) -> dict:
    """Оценка точки: N_SEEDS прогонов, целевое значение — медиана total.

    Взорвавшиеся прогоны (см. BLOWUP_LIMIT) входят в медиану со штрафом
    -BLOWUP_PENALTY; PnL по агентам усредняется только по здоровым прогонам.
    Сиды явные и пишутся в лог/CSV, так что любой прогон воспроизводим.
    При фиксированном SEED все прогоны были бы одинаковы, поэтому делается
    ровно один.
    """
    if SEED is not None:
        seeds = [SEED]
    else:
        seeds = [int.from_bytes(os.urandom(4), "little")
                 for _ in range(N_SEEDS)]
    t0 = time.perf_counter()
    runs = [run_pnl(x, cfg=opt_config(total_steps=TOTAL_STEPS, seed=s),
                    blowup_limit=BLOWUP_LIMIT)
            for s in seeds]
    totals = [-BLOWUP_PENALTY if r["blown"] else r["total"] for r in runs]
    ok = [r for r in runs if not r["blown"]]
    return {
        "pnl": {k: (float(np.mean([r["pnl"][k] for r in ok])) if ok else 0.0)
                for k in AGENT_ORDER},
        "total": float(np.median(totals)),   # целевое значение оптимизации
        "mean": float(np.mean(totals)),
        "std": float(np.std(totals)),
        "n_blown": len(runs) - len(ok),
        "seeds": seeds,
        "elapsed": time.perf_counter() - t0,
    }


def _fmt_params(x) -> str:
    # .4g — 4 значащих цифры: на лог-шкале важен порядок величины
    return "  ".join(f"{n}={v:.4g}" for n, v in zip(PARAM_NAMES, x))


def _fmt_pnl(res: dict) -> str:
    parts = "  ".join(f"{k}={res['pnl'][k]:+8.4f}" for k in AGENT_ORDER)
    return f"{parts}  | median={res['total']:+9.4f}"


class Log:
    """Печать каждого прогона + строка в CSV + лучший результат."""

    def __init__(self, path: str):
        self.i = 0
        self.best_total = -np.inf
        self.best_x = None
        new = not os.path.exists(path)
        self._f = open(path, "a", newline="", encoding="utf-8")
        self._csv = csv.writer(self._f)
        if new:
            self._csv.writerow(("iter",) + PARAM_NAMES + AGENT_ORDER
                               + ("median", "mean", "std", "n_blown",
                                  "seeds", "elapsed_s"))

    def record(self, x, res: dict) -> None:
        self.i += 1
        if res["total"] > self.best_total:
            self.best_total = res["total"]
            self.best_x = list(x)
            star = " <-- новый лучший"
        else:
            star = ""
        print(f"[{self.i:>3}/{N_CALLS}] {_fmt_params(x)}")
        seeds_txt = ",".join(str(s) for s in res["seeds"])
        blown_txt = (f"  ВЗОРВАЛОСЬ {res['n_blown']}/{len(res['seeds'])}"
                     if res["n_blown"] else "")
        print(f"        {_fmt_pnl(res)}  mean={res['mean']:+.4f} "
              f"±{res['std']:.4f}{blown_txt}  "
              f"(seeds={seeds_txt}, {res['elapsed']:.1f}s){star}")
        self._csv.writerow([self.i] + [f"{v:.6g}" for v in x]
                           + [f"{res['pnl'][k]:.6f}" for k in AGENT_ORDER]
                           + [f"{res['total']:.6f}", f"{res['mean']:.6f}",
                              f"{res['std']:.6f}", res["n_blown"],
                              ";".join(str(s) for s in res["seeds"]),
                              f"{res['elapsed']:.2f}"])
        self._f.flush()

    def close(self) -> None:
        self._f.close()


# --------------------------------------------------------------------------- #
# Режимы оптимизации
# --------------------------------------------------------------------------- #

def run_sequential(log: Log):
    """Классический gp_minimize: точка за точкой."""

    def objective(x):
        res = evaluate(x)
        log.record(x, res)
        return -res["total"]

    return gp_minimize(
        objective, SPACE,
        n_calls=N_CALLS,
        n_initial_points=N_INITIAL,
        acq_func="EI",
        random_state=RANDOM_STATE,
        verbose=False,
    )


def run_parallel(log: Log):
    """ask/tell-цикл с батчами по N_WORKERS точек, счёт через joblib."""
    opt = Optimizer(
        SPACE,
        base_estimator="GP",
        n_initial_points=N_INITIAL,
        acq_func="EI",
        random_state=RANDOM_STATE,
    )
    done = 0
    while done < N_CALLS:
        batch = min(N_WORKERS, N_CALLS - done)
        xs = opt.ask(n_points=batch, strategy="cl_min")
        results = Parallel(n_jobs=batch)(delayed(evaluate)(x) for x in xs)
        opt.tell(xs, [-r["total"] for r in results])
        for x, r in zip(xs, results):
            log.record(x, r)
        done += batch
        print(f"    лучший на данный момент: total={log.best_total:+.4f}  "
              f"({_fmt_params(log.best_x)})")
    return opt.get_result()


# --------------------------------------------------------------------------- #

def main() -> None:
    mode = f"параллельный, {N_WORKERS} процессов" if PARALLEL else "последовательный"
    print("=" * 78)
    print(f"gp_minimize: {N_CALLS} прогонов ({N_INITIAL} случайных), режим: {mode}")
    seed_txt = (f"случайные, {N_SEEDS} прогонов на точку (среднее PnL)"
                if SEED is None else f"{SEED} (фиксированный, 1 прогон)")
    print(f"прогон: {TOTAL_STEPS} тиков, сиды: {seed_txt}; лог: {LOG_CSV}")
    print(f"границы: h_m in {H_M_BOUNDS}, h_R in {H_R_BOUNDS} "
          f"(лог-шкала)")
    print("=" * 78)

    log = Log(LOG_CSV)
    t0 = time.perf_counter()
    try:
        result = run_parallel(log) if PARALLEL else run_sequential(log)
    finally:
        log.close()
    elapsed = time.perf_counter() - t0

    best_x, best_total = result.x, -result.fun
    print()
    print("=" * 78)
    print(f"Готово за {elapsed / 60:.1f} мин ({log.i} прогонов)")
    print(f"Лучший суммарный PnL (медиана по сидам): {best_total:+.4f}")
    for name, value in zip(PARAM_NAMES, best_x):
        print(f"    {name} = {value:.4g}")
    # контрольный прогон лучшей точки с печатью PnL по агентам
    res = evaluate(best_x)
    print(f"Контрольный прогон: {_fmt_pnl(res)}")
    print("=" * 78)


if __name__ == "__main__":
    main()
