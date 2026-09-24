"""
param_heatmaps.py — тепловые карты PnL по паре параметров трансляторов.

Для каждого базового набора из BASE_SETS фиксируются все параметры, кроме
двух свипуемых (SWEEPS); пара пробегает лог-сетку N_GRID x N_GRID, в каждой
ячейке — статистика STAT ("mean": среднее total по N_SEEDS прогонам, взорвавшийся
прогон входит как -BLOWUP_PENALTY; "median": медиана по здоровым прогонам).
Сиды ОБЩИЕ для всех ячеек и карт (common random numbers): различия между
ячейками отражают параметры, а не сидовую удачу.

Ячейки, где взорвалось большинство прогонов, закрашиваются серым.

Конфигурация симуляции — из SimConfig по умолчанию (учёт по справедливым
маркам, постоянный риск и выключенный шейдинг арбитражёров, order_size 0.05);
арбитражёры зафиксированы почти идеальными (h_mx = h_my = 1e-3), картируются
только параметры трансляторов. Границы каждой оси — SPAN декад вокруг базовой
точки, обрезанные по AXIS_BOUNDS: карта на 6 декад в обе стороны состояла бы
из взорвавшихся клеток.

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

FIXED_ARBS = dict(h_mx=1e-3, h_my=1e-3)      # почти идеальные арбитражёры

# базовые наборы: карта строится вокруг каждого из них (пик кривой
# flow_scan при order_size = 0.05; после ночной программы см. overnight/README.md)
BASE_SETS = {
    # лучшая проверенная точка ночной программы (20 сидов, 6000 тиков: +42.9 ±8.9,
    # банкротств 0/20); база ночных карт (0.01, 1, 0.01, 1) даёт +37.7 на тех же сидах
    "night_best": dict(h_m1=0.0164, h_r1=0.228, h_m2=0.0061, h_r2=1.64, **FIXED_ARBS),
}

# какие пары параметров свиповать (обе оси — лог-шкала)
SWEEPS = [
    ("h_m2", "h_r2"),   # транслятор тонкой биржи: агрессивность x риск
    ("h_m1", "h_r1"),   # транслятор толстой биржи
    ("h_m1", "h_m2"),   # обе агрессивности (поток через контур)
    ("h_r1", "h_r2"),   # оба риска
]

AXIS_BOUNDS = {"h_m": (1e-4, 10.0), "h_r": (1e-2, 100.0)}   # допустимые границы осей
SPAN = 1.5              # декад вокруг базовой точки в каждую сторону
N_GRID = 15             # точек на ось (карта N_GRID x N_GRID)
N_SEEDS = 3             # прогонов на ячейку, сиды общие для всех
MAP_SEEDS: list[int] | None = None   # None — разыграть при запуске (печатаются)
STAT = "mean"           # "mean" (с штрафом за взрыв) | "median" (по здоровым)

TOTAL_STEPS = 3000
BLOWUP_LIMIT = 1000.0   # |PnL| агента выше — прогон оборван (банкротство)
BLOWUP_PENALTY = 1000.0

PARALLEL = True
N_WORKERS = 12

OUT_DIR = "heatmaps"

PARAM_NAMES = ("h_m1", "h_r1", "h_m2", "h_r2", "h_mx", "h_my")
AGENT_ORDER = ("T1", "T2", "AX", "AY")
PARAM_AGENT = {"h_m1": "T1", "h_r1": "T1", "h_m2": "T2", "h_r2": "T2",
               "h_mx": "AX", "h_my": "AY"}


def axis_bounds(param: str, base_value: float, span: float = SPAN) -> tuple[float, float]:
    """Границы оси: +-span декад вокруг базового значения внутри AXIS_BOUNDS."""
    lo, hi = AXIS_BOUNDS["h_m" if param.startswith("h_m") else "h_r"]
    a = max(lo, base_value / 10 ** span)
    b = min(hi, base_value * 10 ** span)
    return float(a), float(b)


# --------------------------------------------------------------------------- #
# Счёт одной ячейки (top-level — чтобы pickle для joblib работал)
# --------------------------------------------------------------------------- #

def eval_cell(x: list[float], seeds: list[int], total_steps: int,
              blowup_limit: float, stat: str = STAT,
              cfg_overrides: dict | None = None) -> dict:
    """Статистика PnL по сидам в одной точке сетки; NaN — ячейка взорвалась."""
    cfg_overrides = cfg_overrides or {}
    runs = [run_pnl(x, cfg=opt_config(total_steps=total_steps, seed=s, **cfg_overrides),
                    blowup_limit=blowup_limit)
            for s in seeds]
    ok = [r for r in runs if not r["blown"]]
    out = {"n_blown": len(runs) - len(ok)}
    if 2 * len(ok) <= len(runs):          # взорвалось большинство прогонов
        for k in AGENT_ORDER + ("total",):
            out[k] = np.nan
        return out
    if stat == "mean":
        for k in AGENT_ORDER:
            out[k] = float(np.mean([r["pnl"][k] for r in ok]))
        out["total"] = float(np.mean([-BLOWUP_PENALTY if r["blown"] else r["total"]
                                      for r in runs]))
    elif stat == "median":
        for k in AGENT_ORDER:
            out[k] = float(np.median([r["pnl"][k] for r in ok]))
        out["total"] = float(np.median([r["total"] for r in ok]))
    else:
        raise ValueError(f"stat: {stat!r}")
    return out


# --------------------------------------------------------------------------- #
# Построение одной карты
# --------------------------------------------------------------------------- #

def make_heatmap(label: str, base: dict, p1: str, p2: str, seeds: list[int],
                 *, bounds=None, n_grid=N_GRID, total_steps=TOTAL_STEPS,
                 blowup_limit=BLOWUP_LIMIT, stat=STAT, cfg_overrides=None,
                 parallel=PARALLEL, n_workers=N_WORKERS, out_dir=OUT_DIR) -> str:
    """Свип пары (p1, p2) вокруг базового набора; возвращает путь к PNG.

    bounds — ((lo1, hi1), (lo2, hi2)) или None: SPAN декад вокруг base.
    """
    if bounds is None:
        bounds = (axis_bounds(p1, base[p1]), axis_bounds(p2, base[p2]))
    v1 = np.geomspace(bounds[0][0], bounds[0][1], n_grid)
    v2 = np.geomspace(bounds[1][0], bounds[1][1], n_grid)
    cells = []                     # (i, j, вектор 6 параметров)
    for i, a in enumerate(v1):
        for j, b in enumerate(v2):
            p = dict(base)
            p[p1], p[p2] = float(a), float(b)
            cells.append((i, j, [p[name] for name in PARAM_NAMES]))

    t0 = time.perf_counter()
    if parallel:
        results = Parallel(n_jobs=n_workers)(
            delayed(eval_cell)(x, seeds, total_steps, blowup_limit, stat, cfg_overrides)
            for _, _, x in cells)
    else:
        results = [eval_cell(x, seeds, total_steps, blowup_limit, stat, cfg_overrides)
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
    best = None
    if np.isfinite(tot).any():
        i, j = np.unravel_index(np.nanargmax(tot), tot.shape)
        best = (float(v1[i]), float(v2[j]), float(tot[i, j]))
        print(f"  [{label}] {p1} x {p2}: {elapsed:.0f}s, взорвано ячеек "
              f"{int(np.isnan(tot).sum())}/{n_grid ** 2}; max total="
              f"{tot[i, j]:+.4f} при {p1}={v1[i]:.4g}, {p2}={v2[j]:.4g}")

    # --- рисунок ----------------------------------------------------------- #
    def edges(v):
        L = np.log10(v); dL = L[1] - L[0]
        return 10 ** np.linspace(L[0] - dL / 2, L[-1] + dL / 2, len(v) + 1)
    e1, e2 = edges(v1), edges(v2)
    fixed = "  ".join(f"{n}={base[n]:.4g}" for n in PARAM_NAMES if n not in (p1, p2))

    fig, axes = plt.subplots(1, len(panels), figsize=(5.4 * len(panels), 4.8),
                             constrained_layout=True)
    for ax, key in zip(np.atleast_1d(axes), panels):
        data = Z[key]
        m = np.nanmax(np.abs(data)) if np.isfinite(data).any() else 1.0
        cmap = plt.get_cmap("RdYlGn").copy()
        cmap.set_bad("0.82")                     # взорвавшиеся ячейки — серые
        pcm = ax.pcolormesh(e1, e2, data.T, cmap=cmap, vmin=-m, vmax=m, shading="flat")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.plot(base[p1], base[p2], marker="*", ms=14, mec="black", mfc="white", lw=0)
        if best is not None and key == "total":
            ax.plot(best[0], best[1], marker="o", ms=9, mec="black", mfc="none", lw=0)
        ax.set_xlabel(p1); ax.set_ylabel(p2)
        ax.set_title(f"PnL {key}" if key != "total" else f"суммарный PnL ({stat})")
        fig.colorbar(pcm, ax=ax, shrink=0.9)
    fig.suptitle(f"{label}: свип {p1} x {p2}   (зафиксировано: {fixed};  {stat} по "
                 f"{len(seeds)} сидам, {total_steps} тиков)", fontsize=11)

    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, f"{label}_{p1}_{p2}")
    fig.savefig(stem + ".png", dpi=150)
    plt.close(fig)
    np.savez(stem + ".npz", values1=v1, values2=v2, n_blown=n_blown,
             seeds=np.array(seeds), base_names=np.array(PARAM_NAMES),
             base_values=np.array([base[n] for n in PARAM_NAMES]),
             sweep=np.array([p1, p2]), stat=stat, **{k: Z[k] for k in panels})
    return stem + ".png"


# --------------------------------------------------------------------------- #

def main() -> None:
    seeds = (MAP_SEEDS if MAP_SEEDS is not None else
             [int.from_bytes(os.urandom(4), "little") for _ in range(N_SEEDS)])
    n_maps = len(BASE_SETS) * len(SWEEPS)
    print(f"{n_maps} карт по {N_GRID}x{N_GRID} ячеек, {len(seeds)} сидов на "
          f"ячейку (общие для всех карт): {seeds}; статистика {STAT}")
    print(f"оси: +-{SPAN} декад вокруг базы внутри {AXIS_BOUNDS}, прогон {TOTAL_STEPS} тиков"
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


# --------------------------------------------------------------------------- #
# Перерисовка из .npz с обрезанной цветовой шкалой
# --------------------------------------------------------------------------- #

def plot_npz(npz_path: str, png_path: str | None = None, q_neg: float = 0.0) -> str:
    """Перерисовать карту из .npz так, чтобы была видна структура положительной
    области: шкала симметрична относительно нуля с vmax = максимум панели;
    глубокие отрицательные значения (банкротства) насыщаются красным.
    q_neg — нижняя граница шкалы как доля от -vmax (0 -> -vmax)."""
    d = np.load(npz_path)
    v1, v2 = d["values1"], d["values2"]
    p1, p2 = (str(x) for x in d["sweep"])
    base = dict(zip((str(n) for n in d["base_names"]), d["base_values"]))
    stat = str(d["stat"]) if "stat" in d else ""
    panels = [k for k in d.files if k not in ("values1", "values2", "n_blown", "seeds", "base_names",
                                              "base_values", "sweep", "stat")]
    panels = [k for k in panels if k != "total"] + ["total"]

    def edges(v):
        L = np.log10(v); dL = L[1] - L[0]
        return 10 ** np.linspace(L[0] - dL / 2, L[-1] + dL / 2, len(v) + 1)
    e1, e2 = edges(v1), edges(v2)
    fixed = "  ".join(f"{n}={base[n]:.4g}" for n in base if n not in (p1, p2))
    fig, axes = plt.subplots(1, len(panels), figsize=(5.4 * len(panels), 4.8), constrained_layout=True)
    for ax, key in zip(np.atleast_1d(axes), panels):
        data = d[key]
        pos = data[np.isfinite(data) & (data > 0)]
        vmax = float(pos.max()) if pos.size else float(np.nanmax(np.abs(data)) or 1.0)
        cmap = plt.get_cmap("RdYlGn").copy(); cmap.set_bad("0.82")
        pcm = ax.pcolormesh(e1, e2, data.T, cmap=cmap, vmin=-vmax * (1 - q_neg), vmax=vmax, shading="flat")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.plot(base[p1], base[p2], marker="*", ms=14, mec="black", mfc="white", lw=0)
        if np.isfinite(data).any():
            i, j = np.unravel_index(np.nanargmax(data), data.shape)
            ax.plot(v1[i], v2[j], marker="o", ms=9, mec="black", mfc="none", lw=0)
        ax.set_xlabel(p1); ax.set_ylabel(p2)
        ax.set_title(f"PnL {key}" if key != "total" else f"суммарный PnL ({stat}), шкала до +{vmax:.0f}")
        fig.colorbar(pcm, ax=ax, shrink=0.9)
    fig.suptitle(f"свип {p1} x {p2}   (зафиксировано: {fixed}; {len(d['seeds'])} сидов; "
                 f"серое — взорвалось большинство прогонов)", fontsize=11)
    png_path = png_path or npz_path.replace(".npz", "_clip.png")
    fig.savefig(png_path, dpi=150); plt.close(fig)
    return png_path


def plot_totals_grid(npz_paths: list[str], png_path: str, title: str = "") -> str:
    """Одна фигура: панели «суммарный PnL» нескольких карт (2 x N/2)."""
    n = len(npz_paths); cols = 2; rows = (n + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(6.2 * cols, 5.0 * rows), constrained_layout=True)
    for ax, path in zip(np.atleast_1d(axes).ravel(), npz_paths):
        d = np.load(path); v1, v2 = d["values1"], d["values2"]; p1, p2 = (str(x) for x in d["sweep"])
        base = dict(zip((str(x) for x in d["base_names"]), d["base_values"]))
        data = d["total"]; pos = data[np.isfinite(data) & (data > 0)]
        vmax = float(pos.max()) if pos.size else 1.0
        def edges(v):
            L = np.log10(v); dL = L[1] - L[0]
            return 10 ** np.linspace(L[0] - dL / 2, L[-1] + dL / 2, len(v) + 1)
        cmap = plt.get_cmap("RdYlGn").copy(); cmap.set_bad("0.82")
        pcm = ax.pcolormesh(edges(v1), edges(v2), data.T, cmap=cmap, vmin=-vmax, vmax=vmax, shading="flat")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel(p1); ax.set_ylabel(p2)
        ax.plot(base[p1], base[p2], marker="*", ms=14, mec="black", mfc="white", lw=0)
        i, j = np.unravel_index(np.nanargmax(data), data.shape)
        ax.plot(v1[i], v2[j], marker="o", ms=9, mec="black", mfc="none", lw=0)
        ax.set_title(f"{p1} x {p2}: max {data[i, j]:+.1f} при {p1}={v1[i]:.3g}, {p2}={v2[j]:.3g}")
        fig.colorbar(pcm, ax=ax, shrink=0.85)
    for ax in np.atleast_1d(axes).ravel()[n:]:
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=12)
    fig.savefig(png_path, dpi=140); plt.close(fig)
    return png_path
