"""
param_heatmaps.py — тепловые карты PnL по паре параметров одного агента.

Для каждого базового набора из BASE_SETS фиксируются все 6 параметров, кроме
двух свипуемых (SWEEPS); пара пробегает лог-сетку N_GRID x N_GRID, в каждой
ячейке — медиана PnL по N_SEEDS прогонам. Сиды ОБЩИЕ для всех ячеек и карт
(common random numbers): различия между ячейками отражают параметры, а не
сидовую удачу.

Ячейки, где симуляция идёт вразнос (больше половины прогонов упёрлись в
BLOWUP_LIMIT, см. optimize_agents), закрашиваются серым.

На каждую карту рисуется по панели на затронутого агента + суммарный PnL;
PNG и сырые данные (.npz) складываются в OUT_DIR.

Запуск:  python param_heatmaps.py
"""

from __future__ import annotations

import os
import time

import numpy as np
from joblib import Parallel, delayed

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from opt_simulation import opt_config, run_pnl

# --------------------------------------------------------------------------- #
# Настройки
# --------------------------------------------------------------------------- #

# базовые наборы: карта строится вокруг каждого из них
BASE_SETS = {
    "baseline": dict(h_m1=1.0, h_r1=1.0, h_m2=1.0, h_r2=1.0,
                     h_mx=1.0, h_my=1.0),
    # лучшая устойчивая точка из opt_log.csv (median=+68.1, mean=+68.4,
    # 0 взрывов из 10 сидов)
    "opt_best": dict(h_m1=0.263, h_r1=0.0646, h_m2=0.02636, h_r2=4.301,
                     h_mx=0.01115, h_my=0.03708),
}

# какие пары параметров свиповать (обе оси — лог-шкала)
SWEEPS = [
    ("h_m1", "h_r1"),   # параметры T1
    ("h_m2", "h_r2"),   # параметры T2
    ("h_mx", "h_my"),   # интенсивности арбитражёров AX и AY
]

BOUNDS = (1e-4, 100.0)  # диапазон каждой свипуемой оси
N_GRID = 15             # точек на ось (карта N_GRID x N_GRID)
N_SEEDS = 3             # прогонов на ячейку (медиана), сиды общие для всех
MAP_SEEDS: list[int] | None = None   # None — разыграть при запуске (печатаются)

TOTAL_STEPS = 6000
BLOWUP_LIMIT = 1000.0   # как в optimize_agents: |PnL| выше — прогон оборван

PARALLEL = True
N_WORKERS = 12

OUT_DIR = "heatmaps"

PARAM_NAMES = ("h_m1", "h_r1", "h_m2", "h_r2", "h_mx", "h_my")
AGENT_ORDER = ("T1", "T2", "AX", "AY")
PARAM_AGENT = {"h_m1": "T1", "h_r1": "T1", "h_m2": "T2", "h_r2": "T2",
               "h_mx": "AX", "h_my": "AY"}


# --------------------------------------------------------------------------- #
# Счёт одной ячейки (top-level — чтобы pickle для joblib работал)
# --------------------------------------------------------------------------- #

def eval_cell(x: list[float], seeds: list[int], total_steps: int,
              blowup_limit: float) -> dict:
    """Медианы PnL по сидам в одной точке сетки; NaN — ячейка взорвалась."""
    runs = [run_pnl(x, cfg=opt_config(total_steps=total_steps, seed=s),
                    blowup_limit=blowup_limit)
            for s in seeds]
    ok = [r for r in runs if not r["blown"]]
    out = {"n_blown": len(runs) - len(ok)}
    if 2 * len(ok) <= len(runs):          # взорвалось большинство прогонов
        for k in AGENT_ORDER + ("total",):
            out[k] = np.nan
    else:
        for k in AGENT_ORDER:
            out[k] = float(np.median([r["pnl"][k] for r in ok]))
        out["total"] = float(np.median([r["total"] for r in ok]))
    return out


# --------------------------------------------------------------------------- #
# Построение одной карты
# --------------------------------------------------------------------------- #

def make_heatmap(label: str, base: dict, p1: str, p2: str, seeds: list[int],
                 *, bounds=BOUNDS, n_grid=N_GRID, total_steps=TOTAL_STEPS,
                 blowup_limit=BLOWUP_LIMIT, parallel=PARALLEL,
                 n_workers=N_WORKERS, out_dir=OUT_DIR) -> str:
    """Свип пары (p1, p2) вокруг базового набора; возвращает путь к PNG."""
    values = np.geomspace(bounds[0], bounds[1], n_grid)
    cells = []                     # (i, j, вектор 6 параметров)
    for i, v1 in enumerate(values):
        for j, v2 in enumerate(values):
            p = dict(base)
            p[p1], p[p2] = float(v1), float(v2)
            cells.append((i, j, [p[name] for name in PARAM_NAMES]))

    t0 = time.perf_counter()
    if parallel:
        results = Parallel(n_jobs=n_workers)(
            delayed(eval_cell)(x, seeds, total_steps, blowup_limit)
            for _, _, x in cells)
    else:
        results = [eval_cell(x, seeds, total_steps, blowup_limit)
                   for _, _, x in cells]
    elapsed = time.perf_counter() - t0

    panels = list(dict.fromkeys([PARAM_AGENT[p1], PARAM_AGENT[p2]])) + ["total"]
    Z = {k: np.full((n_grid, n_grid), np.nan) for k in panels}
    n_blown = np.zeros((n_grid, n_grid), int)
    for (i, j, _), res in zip(cells, results):
        for k in panels:
            Z[k][i, j] = res[k]
        n_blown[i, j] = res["n_blown"]

    tot = Z["total"]
    if np.isfinite(tot).any():
        i, j = np.unravel_index(np.nanargmax(tot), tot.shape)
        print(f"  [{label}] {p1} x {p2}: {elapsed:.0f}s, взорвано ячеек "
              f"{int(np.isnan(tot).sum())}/{n_grid ** 2}; max total="
              f"{tot[i, j]:+.4f} при {p1}={values[i]:.4g}, {p2}={values[j]:.4g}")

    # --- рисунок ----------------------------------------------------------- #
    L = np.log10(values)
    dL = L[1] - L[0]
    edges = 10 ** np.linspace(L[0] - dL / 2, L[-1] + dL / 2, n_grid + 1)
    fixed = "  ".join(f"{n}={base[n]:.4g}" for n in PARAM_NAMES
                      if n not in (p1, p2))

    fig, axes = plt.subplots(1, len(panels),
                             figsize=(5.4 * len(panels), 4.8),
                             constrained_layout=True)
    for ax, key in zip(np.atleast_1d(axes), panels):
        data = Z[key]
        m = np.nanmax(np.abs(data)) if np.isfinite(data).any() else 1.0
        cmap = plt.get_cmap("RdYlGn").copy()
        cmap.set_bad("0.82")                     # взорвавшиеся ячейки — серые
        pcm = ax.pcolormesh(edges, edges, data.T, cmap=cmap,
                            vmin=-m, vmax=m, shading="flat")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.plot(base[p1], base[p2], marker="*", ms=14, mec="black",
                mfc="white", lw=0)              # базовая точка
        ax.set_xlabel(p1)
        ax.set_ylabel(p2)
        ax.set_title(f"PnL {key}" if key != "total" else "суммарный PnL")
        fig.colorbar(pcm, ax=ax, shrink=0.9)
    fig.suptitle(f"{label}: свип {p1} x {p2}   (зафиксировано: {fixed};  "
                 f"медиана по {len(seeds)} сидам)", fontsize=11)

    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, f"{label}_{p1}_{p2}")
    fig.savefig(stem + ".png", dpi=150)
    plt.close(fig)
    np.savez(stem + ".npz", values=values, n_blown=n_blown,
             seeds=np.array(seeds),
             base_names=np.array(PARAM_NAMES),
             base_values=np.array([base[n] for n in PARAM_NAMES]),
             sweep=np.array([p1, p2]),
             **{k: Z[k] for k in panels})
    return stem + ".png"


# --------------------------------------------------------------------------- #

def main() -> None:
    seeds = (MAP_SEEDS if MAP_SEEDS is not None else
             [int.from_bytes(os.urandom(4), "little") for _ in range(N_SEEDS)])
    n_maps = len(BASE_SETS) * len(SWEEPS)
    print(f"{n_maps} карт по {N_GRID}x{N_GRID} ячеек, {len(seeds)} сидов на "
          f"ячейку (общие для всех карт): {seeds}")
    print(f"оси: {BOUNDS[0]:g}..{BOUNDS[1]:g} (лог), прогон {TOTAL_STEPS} тиков"
          + (f", {N_WORKERS} процессов" if PARALLEL else ", последовательно"))

    paths = []
    for label, base in BASE_SETS.items():
        for p1, p2 in SWEEPS:
            paths.append(make_heatmap(label, base, p1, p2, seeds))
    print("\nГотово:")
    for p in paths:
        print(" ", p)


if __name__ == "__main__":
    main()
