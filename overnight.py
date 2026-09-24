"""
overnight.py — ночная программа: есть ли у задачи нормальный оптимум?

Конфигурация: SimConfig по умолчанию (order_size 0.05, справедливые марки,
постоянный риск и выключенный шейдинг арбитражёров), арбитражёры почти
идеальные (h_mx = h_my = 1e-3), оптимизируются 4 параметра трансляторов.
Критерий: среднее по сидам, банкротство = -1000.

Этапы (каждый защищён try/except; результаты пишутся в OUT сразу, README
пересобирается после каждого этапа):
    1. flow    — кривая PnL(s) при общем множителе s для h_m1, h_m2; T=3000 и 6000
    2. gp      — N_GP независимых гауссовых оптимизаций (разные random_state и
                 наборы сидов), затем проверка лучших точек на 20 свежих сидах
    3. base    — выбор базовой точки для карт: лучшая проверенная, не на границе,
                 без банкротств; при разногласии GP — точка с кривой flow
    4. map1    — карта h_m2 x h_r2 вокруг базы (15x15, 3 сида)
    5. econ    — сдвиг пика при изменении среды: order_size, arrival_rate тонкой
                 биржи, price_std (проверка: оптимум = доля ликвидности биржи)
    6. robust  — 20 сидов в базе на горизонтах 1500..12000; траектории PnL
    7. maps    — остальные карты, пока есть время
Дедлайн: 07:00 локального времени; перед долгими этапами проверяется остаток.

Запуск:  python overnight.py [--smoke] [--out DIR] [--deadline HH:MM]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
import traceback

import numpy as np
from joblib import Parallel, delayed

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from skopt import Optimizer
from skopt.space import Real

import flow_scan
import param_heatmaps as PH
from coupled_market import ExchangeConfig
from opt_simulation import PARAM_NAMES, opt_config, run_pnl

AGENTS = ("T1", "T2", "AX", "AY")
OPT_NAMES = ("h_m1", "h_r1", "h_m2", "h_r2")
FIXED_ARBS = dict(h_mx=1e-3, h_my=1e-3)
H_M_BOUNDS = (1e-4, 10.0)
H_R_BOUNDS = (1e-2, 100.0)
PENALTY = 1000.0
N_JOBS = 12


# --------------------------------------------------------------------------- #
# Инфраструктура
# --------------------------------------------------------------------------- #

class Night:
    def __init__(self, out: str, deadline: dt.datetime, smoke: bool):
        self.out = out
        os.makedirs(out, exist_ok=True)
        os.makedirs(os.path.join(out, "heatmaps"), exist_ok=True)
        self.deadline = deadline
        self.smoke = smoke
        self.res: dict = {"config": {"smoke": smoke, "deadline": deadline.isoformat()},
                          "stages": {}}
        self.t_start = time.perf_counter()
        self.rate = None          # прогонов (3000 тиков) в секунду, измеряется
        self.log_path = os.path.join(out, "log.txt")
        self.log(f"старт {dt.datetime.now():%Y-%m-%d %H:%M}, дедлайн {deadline:%H:%M}, smoke={smoke}")

    # --- служебное -------------------------------------------------------- #
    def log(self, msg: str) -> None:
        line = f"[{dt.datetime.now():%H:%M:%S}] {msg}"
        print(line, flush=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def remaining_min(self) -> float:
        return (self.deadline - dt.datetime.now()).total_seconds() / 60.0

    def est_min(self, n_runs: int, steps: int = 3000) -> float:
        rate = self.rate or 0.25          # запасная оценка: 0.25 прогона/с
        return n_runs * (steps + 200) / 3200 / rate / 60.0

    def save(self) -> None:
        with open(os.path.join(self.out, "results.json"), "w", encoding="utf-8") as f:
            json.dump(self.res, f, indent=1, ensure_ascii=False, default=float)
        write_readme(self)

    def stage(self, name: str, fn) -> None:
        self.log(f"--- этап {name}: старт (осталось {self.remaining_min():.0f} мин)")
        t0 = time.perf_counter()
        try:
            fn()
            self.res["stages"][name] = self.res["stages"].get(name, {})
            self.res["stages"][name]["ok"] = True
        except Exception:
            tb = traceback.format_exc()
            self.log(f"этап {name}: ОШИБКА\n{tb}")
            self.res["stages"].setdefault(name, {})["error"] = tb
        self.res["stages"].setdefault(name, {})["minutes"] = (time.perf_counter() - t0) / 60.0
        self.save()
        self.log(f"--- этап {name}: конец ({(time.perf_counter() - t0) / 60:.1f} мин)")

    # --- размеры --------------------------------------------------------- #
    @property
    def S(self) -> dict:
        if self.smoke:
            return dict(steps=150, steps_long=300, flow_seeds=1, flow_scales=[0.03, 0.01, 0.003],
                        flow_scales_long=[0.02, 0.007], n_gp=1, gp_calls=4, gp_init=3, gp_batch=3,
                        gp_seeds=1, verify_seeds=2, grid=3, map_seeds=1, econ_scales=[1.5, 1.0, 0.6],
                        econ_seeds=1, robust_seeds=2, robust_T=[150, 300], path_seeds=2)
        return dict(steps=3000, steps_long=6000, flow_seeds=6,
                    flow_scales=[0.05, 0.03, 0.02, 0.015, 0.01, 0.007, 0.005, 0.003],
                    flow_scales_long=[0.03, 0.02, 0.015, 0.01, 0.007, 0.005],
                    n_gp=3, gp_calls=80, gp_init=16, gp_batch=12, gp_seeds=4, verify_seeds=20,
                    grid=15, map_seeds=3, econ_scales=[3.0, 2.0, 1.4, 1.0, 0.7, 0.5, 0.35, 0.25],
                    econ_seeds=4, robust_seeds=20, robust_T=[1500, 3000, 6000, 12000], path_seeds=6)


def theta6(h_m1, h_r1, h_m2, h_r2) -> list[float]:
    d = dict(h_m1=h_m1, h_r1=h_r1, h_m2=h_m2, h_r2=h_r2, **FIXED_ARBS)
    return [float(d[n]) for n in PARAM_NAMES]


def evaluate_points(points, seeds, steps, n_jobs=N_JOBS, overrides=None) -> list[dict]:
    """Список 6-векторов x сиды -> сводка по каждой точке (mean с штрафом)."""
    overrides = overrides or {}
    jobs = [(k, s) for k in range(len(points)) for s in seeds]
    runs = Parallel(n_jobs=min(n_jobs, len(jobs)))(
        delayed(run_pnl)(list(points[k]), cfg=opt_config(total_steps=steps, seed=s, **overrides),
                         blowup_limit=PENALTY) for k, s in jobs)
    out = []
    for k in range(len(points)):
        rr = [r for (kk, _), r in zip(jobs, runs) if kk == k]
        tot = [(-PENALTY if r["blown"] else r["total"]) for r in rr]
        ok = [r for r in rr if not r["blown"]]
        out.append({"theta": list(map(float, points[k])), "mean": float(np.mean(tot)),
                    "median": float(np.median(tot)), "std": float(np.std(tot)),
                    "n_blown": len(rr) - len(ok), "n": len(rr), "totals": tot,
                    "pnl": {a: (float(np.mean([r["pnl"][a] for r in ok])) if ok else 0.0)
                            for a in AGENTS}})
    return out


# --------------------------------------------------------------------------- #
# Этап 1: кривая по скорости
# --------------------------------------------------------------------------- #

def stage_flow(N: Night) -> None:
    S = N.S
    base = theta6(1.0, 1.0, 1.0, 1.0)
    idx = (0, 2)                                     # h_m1, h_m2
    seeds = [101 + i for i in range(S["flow_seeds"])]
    t0 = time.perf_counter()
    rows, el = flow_scan.scan(base, idx, S["flow_scales"], seeds, S["steps"], True, True, {}, N_JOBS)
    n_runs = len(S["flow_scales"]) * len(seeds)
    N.rate = n_runs / el * (S["steps"] + 200) / 3200
    N.log(f"скорость счёта: {N.rate:.3f} прогона(3000 тиков)/с")
    rows_long, _ = flow_scan.scan(base, idx, S["flow_scales_long"],
                                  seeds[:max(1, S["flow_seeds"] * 2 // 3)], S["steps_long"],
                                  True, True, {}, N_JOBS)
    best = max(rows, key=lambda r: r["mean"])
    best_long = max(rows_long, key=lambda r: r["mean"])
    N.res["stages"]["flow"] = {"rows": rows, "rows_long": rows_long, "verdict": flow_scan.verdict(rows),
                               "s_star": best["s"], "s_star_long": best_long["s"],
                               "peak_mean": best["mean"], "peak_mean_long": best_long["mean"]}
    # рисунок
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for rr, lab in ((rows, f"T={S['steps']}"), (rows_long, f"T={S['steps_long']}")):
        ax.errorbar([1 / r["s"] for r in rr], [r["mean"] for r in rr],
                    yerr=[r["std"] / np.sqrt(r["n"]) for r in rr], marker="o", label=lab)
    ax.set_xscale("log"); ax.set_xlabel("скорость оборота 1/s  (h_m1 = h_m2 = s)")
    ax.set_ylabel("средний PnL (банкротство = -1000)"); ax.grid(alpha=0.3); ax.legend()
    ax.set_title("Кривая прибыли по скорости, идеальные арбитражёры")
    fig.tight_layout(); fig.savefig(os.path.join(N.out, "flow_curve.png"), dpi=140); plt.close(fig)


# --------------------------------------------------------------------------- #
# Этап 2: несколько гауссовых оптимизаций + проверка
# --------------------------------------------------------------------------- #

def stage_gp(N: Night) -> None:
    S = N.S
    space = [Real(*H_M_BOUNDS, prior="log-uniform", name="h_m1"),
             Real(*H_R_BOUNDS, prior="log-uniform", name="h_r1"),
             Real(*H_M_BOUNDS, prior="log-uniform", name="h_m2"),
             Real(*H_R_BOUNDS, prior="log-uniform", name="h_r2")]
    runs_out = []
    N.res["stages"]["gp"] = {"runs": runs_out}
    reserve = 0.0 if N.smoke else 150.0          # минут на остальные этапы
    for k in range(1, S["n_gp"] + 1):
        need = N.est_min(S["gp_calls"] * S["gp_seeds"], S["steps"])
        if N.remaining_min() < need + reserve:
            N.log(f"GP #{k}: пропуск, нужно {need:.0f} мин, осталось {N.remaining_min():.0f}")
            break
        seeds = [300 * k + i for i in range(1, S["gp_seeds"] + 1)]     # CRN внутри прогона
        opt = Optimizer(space, base_estimator="GP", n_initial_points=S["gp_init"],
                        acq_func="EI", random_state=k)
        hist = []
        done = 0
        while done < S["gp_calls"]:
            batch = min(S["gp_batch"], S["gp_calls"] - done)
            xs = opt.ask(n_points=batch, strategy="cl_min")
            pts = [theta6(*x) for x in xs]
            ev = evaluate_points(pts, seeds, S["steps"])
            opt.tell(xs, [-e["mean"] for e in ev])
            for x, e in zip(xs, ev):
                hist.append({"x": [float(v) for v in x], "mean": e["mean"], "n_blown": e["n_blown"]})
            done += batch
            b = max(hist, key=lambda h: h["mean"])
            N.log(f"GP #{k}: {done}/{S['gp_calls']}, лучший mean={b['mean']:+.2f} при "
                  + " ".join(f"{n}={v:.3g}" for n, v in zip(OPT_NAMES, b["x"])))
        best = max(hist, key=lambda h: h["mean"])
        runs_out.append({"random_state": k, "seeds": seeds, "best": best, "history": hist})
        N.res["stages"]["gp"] = {"runs": runs_out}
        N.save()
    # проверка кандидатов на свежих сидах, длинный горизонт
    cands = [{"name": f"GP #{r['random_state']}", "theta": theta6(*r["best"]["x"])} for r in runs_out]
    s_star = N.res["stages"].get("flow", {}).get("s_star", 0.01)
    cands.append({"name": "flow s*", "theta": theta6(s_star, 1.0, s_star, 1.0)})
    vseeds = [401 + i for i in range(S["verify_seeds"])]
    ev = evaluate_points([c["theta"] for c in cands], vseeds, S["steps_long"])
    for c, e in zip(cands, ev):
        c.update({k: e[k] for k in ("mean", "median", "std", "n_blown", "n", "pnl", "totals")})
    N.res["stages"]["gp"]["candidates"] = cands
    N.res["stages"]["gp"]["verify_seeds"] = vseeds
    N.res["stages"]["gp"]["verify_steps"] = S["steps_long"]
    for c in cands:
        N.log(f"проверка {c['name']}: mean={c['mean']:+.2f} ±{c['std']:.1f}, банкротств {c['n_blown']}/{c['n']}  "
              + " ".join(f"{n}={v:.3g}" for n, v in zip(PARAM_NAMES, c["theta"])))


# --------------------------------------------------------------------------- #
# Этап 3: выбор базы
# --------------------------------------------------------------------------- #

def on_boundary(theta) -> list[str]:
    bad = []
    for name, v in zip(PARAM_NAMES, theta):
        if name in FIXED_ARBS:
            continue
        lo, hi = H_M_BOUNDS if name.startswith("h_m") else H_R_BOUNDS
        if v <= lo * 2 or v >= hi / 2:
            bad.append(name)
    return bad


def stage_base(N: Night) -> None:
    gp = N.res["stages"].get("gp", {})
    cands = gp.get("candidates", [])
    notes = []
    s_star = N.res["stages"].get("flow", {}).get("s_star", 0.01)
    fallback = theta6(s_star, 1.0, s_star, 1.0)
    good = []
    for c in cands:
        b = on_boundary(c["theta"])
        frac_bl = c["n_blown"] / max(c["n"], 1)
        if b:
            notes.append(f"{c['name']}: на границе по {b}")
        elif frac_bl > 0.1:
            notes.append(f"{c['name']}: банкротств {c['n_blown']}/{c['n']}")
        else:
            good.append(c)
    gp_runs = gp.get("runs", [])
    if len(gp_runs) >= 2:
        xs = np.log10([r["best"]["x"] for r in gp_runs])
        spread = xs.max(0) - xs.min(0)
        notes.append("разброс лучших точек GP по прогонам (декад): "
                     + " ".join(f"{n}={v:.1f}" for n, v in zip(OPT_NAMES, spread)))
        if (spread > 1.0).any():
            notes.append("GP не сходится в одну точку: отличия больше декады — карты строятся "
                         "вокруг лучшей проверенной точки, а не вокруг одного прогона GP")
    if good:
        best = max(good, key=lambda c: c["mean"])
        base = best["theta"]; src = best["name"]
    else:
        base = fallback; src = "flow s* (все кандидаты GP на границе или с банкротствами)"
    base_d = dict(zip(PARAM_NAMES, base))
    N.res["stages"]["base"] = {"theta": base_d, "source": src, "notes": notes}
    N.log(f"база для карт: {src}: " + " ".join(f"{n}={v:.3g}" for n, v in base_d.items()))
    for n in notes:
        N.log("  " + n)


def map_bounds(base_d: dict, p: str) -> tuple[float, float]:
    return PH.axis_bounds(p, base_d[p], span=1.5)


def make_map(N: Night, p1: str, p2: str) -> None:
    S = N.S
    base_d = N.res["stages"]["base"]["theta"]
    seeds = [601 + i for i in range(S["map_seeds"])]
    bounds = (map_bounds(base_d, p1), map_bounds(base_d, p2))
    N.log(f"карта {p1} x {p2}: границы {p1} {bounds[0]}, {p2} {bounds[1]}")
    path = PH.make_heatmap("night", base_d, p1, p2, seeds, bounds=bounds, n_grid=S["grid"],
                           total_steps=S["steps"], stat="mean", n_workers=N_JOBS,
                           out_dir=os.path.join(N.out, "heatmaps"))
    d = np.load(path.replace(".png", ".npz"))
    tot = d["total"]
    finite = np.isfinite(tot)
    info = {"png": os.path.relpath(path, N.out), "blown_cells": int((~finite).sum()),
            "cells": int(tot.size)}
    if finite.any():
        i, j = np.unravel_index(np.nanargmax(tot), tot.shape)
        info.update({"best": {p1: float(d["values1"][i]), p2: float(d["values2"][j]),
                              "total": float(tot[i, j])},
                     "edge_best": bool(i in (0, tot.shape[0] - 1) or j in (0, tot.shape[1] - 1))})
    N.res["stages"].setdefault("maps", {})[f"{p1}_x_{p2}"] = info
    N.save()


def stage_map1(N: Night) -> None:
    make_map(N, "h_m2", "h_r2")


def stage_maps(N: Night) -> None:
    S = N.S
    for p1, p2 in (("h_m1", "h_r1"), ("h_m1", "h_m2"), ("h_r1", "h_r2")):
        need = N.est_min(S["grid"] ** 2 * S["map_seeds"], S["steps"])
        if N.remaining_min() < need + 10:
            N.log(f"карта {p1} x {p2}: пропуск, нужно {need:.0f} мин, осталось {N.remaining_min():.0f}")
            continue
        make_map(N, p1, p2)


# --------------------------------------------------------------------------- #
# Этап 5: экономика оптимума — сдвиг пика при изменении среды
# --------------------------------------------------------------------------- #

def venue_cfgs(order_size=0.05, arrival2=3.0, price_std=10.0) -> dict:
    return dict(
        venue1=ExchangeConfig(name="1", arrival_rate=10.0, order_size=order_size, order_ttl=80,
                              price_std=price_std, ewma_half_life=10.0),
        venue2=ExchangeConfig(name="2", arrival_rate=arrival2, order_size=order_size, order_ttl=80,
                              price_std=price_std, ewma_half_life=10.0))


def stage_econ(N: Night) -> None:
    S = N.S
    base_d = N.res["stages"]["base"]["theta"]
    base = [base_d[n] for n in PARAM_NAMES]
    idx = (0, 2)
    seeds = [701 + i for i in range(S["econ_seeds"])]
    variants = []
    for os_ in ((0.02, 0.05, 0.1, 0.2) if not N.smoke else (0.05,)):
        variants.append(("order_size", os_, venue_cfgs(order_size=os_), os_ / 0.05))
    for ar in ((1.5, 6.0) if not N.smoke else ()):
        variants.append(("arrival_rate2", ar, venue_cfgs(arrival2=ar), ar / 3.0))
    for ps in ((5.0, 20.0) if not N.smoke else ()):
        variants.append(("price_std", ps, venue_cfgs(price_std=ps), 1.0))
    out = []
    for kind, val, ovr, center in variants:
        need = N.est_min(len(S["econ_scales"]) * len(seeds), S["steps"])
        if N.remaining_min() < need + 30:
            N.log(f"econ {kind}={val}: пропуск (осталось {N.remaining_min():.0f} мин)")
            continue
        scales = [s * center for s in S["econ_scales"]]     # центр сдвинут по предсказанию
        rows, _ = flow_scan.scan(base, idx, scales, seeds, S["steps"], True, True, ovr, N_JOBS)
        best = max(rows, key=lambda r: r["mean"])
        liq = (val if kind == "arrival_rate2" else 3.0) * (val if kind == "order_size" else 0.05)
        rec = {"kind": kind, "value": val, "rows": rows, "peak": best,
               "arriving_units_per_tick": liq,
               "hedge_share_at_peak": best["levels2"] * (val if kind == "order_size" else 0.05) / liq}
        out.append(rec)
        N.log(f"econ {kind}={val}: пик s={best['s']:.4g} mean={best['mean']:+.1f}, "
              f"хедж T2 = {rec['hedge_share_at_peak']*100:.1f}% приходящего объёма, уступка {best['conc2']*100:.2f}%")
        N.res["stages"]["econ"] = {"variants": out}
        N.save()
    # рисунок
    kinds = sorted({v["kind"] for v in out})
    if kinds:
        fig, axes = plt.subplots(1, len(kinds), figsize=(6 * len(kinds), 4.5), squeeze=False)
        for ax, kind in zip(axes[0], kinds):
            for v in [v for v in out if v["kind"] == kind]:
                ax.plot([r["flow"] for r in v["rows"]], [r["mean"] for r in v["rows"]], marker="o",
                        label=f"{kind}={v['value']}")
            ax.set_xscale("log"); ax.set_xlabel("поток через контур, X1/тик"); ax.set_ylabel("средний PnL")
            ax.grid(alpha=0.3); ax.legend(); ax.set_title(f"сдвиг пика: {kind}")
        fig.tight_layout(); fig.savefig(os.path.join(N.out, "econ_curves.png"), dpi=140); plt.close(fig)


# --------------------------------------------------------------------------- #
# Этап 6: робастность в базе
# --------------------------------------------------------------------------- #

def stage_robust(N: Night) -> None:
    S = N.S
    base_d = N.res["stages"]["base"]["theta"]
    base = [base_d[n] for n in PARAM_NAMES]
    seeds = [801 + i for i in range(S["robust_seeds"])]
    byT = {}
    for T in S["robust_T"]:
        need = N.est_min(len(seeds), T)
        if N.remaining_min() < need + 20:
            N.log(f"robust T={T}: пропуск"); continue
        e = evaluate_points([base], seeds, T)[0]
        byT[T] = e
        N.log(f"robust T={T}: mean={e['mean']:+.2f} median={e['median']:+.2f} std={e['std']:.1f} "
              f"банкротств {e['n_blown']}/{e['n']}")
    N.res["stages"]["robust"] = {"byT": {str(k): v for k, v in byT.items()}, "seeds": seeds}
    N.save()
    Ts = sorted(byT)
    if Ts:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        axes[0].errorbar(Ts, [byT[T]["mean"] for T in Ts], yerr=[byT[T]["std"] / np.sqrt(byT[T]["n"]) for T in Ts],
                         marker="o"); axes[0].set_xscale("log"); axes[0].set_xlabel("тиков T")
        axes[0].set_ylabel("средний PnL"); axes[0].grid(alpha=0.3); axes[0].set_title("масштабирование по горизонту")
        Tm = max(Ts)
        axes[1].hist(byT[Tm]["totals"], bins=12, color="tab:green", alpha=0.8)
        axes[1].set_title(f"распределение PnL по {byT[Tm]['n']} сидам, T={Tm}")
        axes[1].set_xlabel("PnL"); axes[1].grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(N.out, "robust.png"), dpi=140); plt.close(fig)
    # траектории PnL по агентам
    T = S["steps_long"]
    pseeds = [901 + i for i in range(S["path_seeds"])]
    runs = Parallel(n_jobs=min(N_JOBS, len(pseeds)))(
        delayed(run_pnl)(base, cfg=opt_config(total_steps=T, seed=s), blowup_limit=PENALTY,
                         return_equity=True) for s in pseeds)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for r in runs:
        eq = r["equity"][:, :r["end_tick"] + 1]
        axes[0].plot(eq.sum(0), lw=1)
    axes[0].set_title("суммарный PnL по времени, разные сиды"); axes[0].set_xlabel("тик"); axes[0].grid(alpha=0.3)
    eq = runs[0]["equity"][:, :runs[0]["end_tick"] + 1]
    for i, a in enumerate(AGENTS):
        axes[1].plot(eq[i], lw=1, label=a)
    axes[1].legend(); axes[1].set_title(f"PnL по агентам, сид {pseeds[0]}"); axes[1].set_xlabel("тик"); axes[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(N.out, "paths.png"), dpi=140); plt.close(fig)


# --------------------------------------------------------------------------- #
# README
# --------------------------------------------------------------------------- #

def _fmt_theta(d: dict) -> str:
    return ", ".join(f"{n} = {d[n]:.3g}" for n in PARAM_NAMES)


def write_readme(N: Night) -> None:
    R = N.res["stages"]
    L = ["# Ночная программа: есть ли нормальный оптимум?", "",
         f"Старт {N.res['config']['deadline'][:10]}, дедлайн {N.res['config']['deadline'][11:16]}; "
         f"конфигурация: SimConfig по умолчанию (order_size 0.05, справедливые марки, арбитражёры без "
         f"шейдинга, риск арбитражёров — константа), h_mx = h_my = 1e-3 зафиксированы, критерий — "
         f"среднее по сидам с банкротством как −1000. Подробный лог: `log.txt`, сырые данные: `results.json`.", ""]
    st = R.get("flow", {})
    if "rows" in st:
        L += ["## 1. Кривая прибыли по скорости (`flow_curve.png`)", "",
              "| s (h_m1 = h_m2) | mean | median | std | банкр. | поток X1/тик | уровней книги 2 за тик | уступка хеджа 2 |",
              "|---|---|---|---|---|---|---|---|"]
        for r in st["rows"]:
            L.append(f"| {r['s']:g} | {r['mean']:+.1f} | {r['median']:+.1f} | {r['std']:.1f} | {r['bankrupt']}/{r['n']} | "
                     f"{r['flow']:.3f} | {r['levels2']:.3f} | {r['conc2']*100:.2f}% |")
        L += ["", f"Вердикт стенда: {st['verdict']}", "",
              f"Пик при T=3000: s* = {st['s_star']:g} (mean {st['peak_mean']:+.1f}); при T=6000: "
              f"s* = {st['s_star_long']:g} (mean {st['peak_mean_long']:+.1f}).", ""]
    st = R.get("gp", {})
    if "runs" in st:
        L += ["## 2. Гауссова оптимизация (независимые прогоны)", "",
              "| прогон | лучшая точка (по своим сидам) | mean |", "|---|---|---|"]
        for r in st["runs"]:
            L.append(f"| GP #{r['random_state']} (сиды {r['seeds']}) | "
                     + ", ".join(f"{n} = {v:.3g}" for n, v in zip(OPT_NAMES, r["best"]["x"]))
                     + f" | {r['best']['mean']:+.1f} |")
        if "candidates" in st:
            L += ["", f"Проверка кандидатов на {len(st['verify_seeds'])} свежих сидах, T = {st['verify_steps']}:", "",
                  "| кандидат | параметры | mean | median | std | банкр. | T1 | T2 | AX | AY |", "|---|---|---|---|---|---|---|---|---|---|"]
            for c in st["candidates"]:
                d = dict(zip(PARAM_NAMES, c["theta"]))
                L.append(f"| {c['name']} | {', '.join(f'{n}={d[n]:.3g}' for n in OPT_NAMES)} | {c['mean']:+.1f} | "
                         f"{c['median']:+.1f} | {c['std']:.1f} | {c['n_blown']}/{c['n']} | "
                         + " | ".join(f"{c['pnl'][a]:+.1f}" for a in AGENTS) + " |")
        L.append("")
    st = R.get("base", {})
    if "theta" in st:
        L += ["## 3. База для карт", "", f"**{st['source']}**: {_fmt_theta(st['theta'])}", ""]
        L += [f"- {n}" for n in st.get("notes", [])] + [""]
    st = R.get("maps", {})
    if st:
        L += ["## 4. Тепловые карты (`heatmaps/`)", "", "| карта | файл | взорванных ячеек | лучшая ячейка | лучшая на краю? |",
              "|---|---|---|---|---|"]
        for k, v in st.items():
            if not isinstance(v, dict) or "png" not in v:
                continue
            b = v.get("best", {})
            L.append(f"| {k} | `{v['png']}` | {v['blown_cells']}/{v['cells']} | "
                     + (", ".join(f"{kk}={vv:.3g}" for kk, vv in b.items()) if b else "—")
                     + f" | {'да' if v.get('edge_best') else 'нет'} |")
        L.append("")
    st = R.get("econ", {})
    if st.get("variants"):
        L += ["## 5. Экономика оптимума: сдвиг пика при изменении среды (`econ_curves.png`)", "",
              "| параметр среды | значение | пик: s | пик: mean | поток X1/тик | хедж T2, % приходящего объёма | уступка |",
              "|---|---|---|---|---|---|---|"]
        for v in st["variants"]:
            p = v["peak"]
            L.append(f"| {v['kind']} | {v['value']:g} | {p['s']:.4g} | {p['mean']:+.1f} | {p['flow']:.3f} | "
                     f"{v['hedge_share_at_peak']*100:.1f}% | {p['conc2']*100:.2f}% |")
        L += ["", "Если доля приходящего объёма в пике примерно одинакова по строкам, оптимум задан "
              "ликвидностью тонкой биржи, а не константами агентов.", ""]
    st = R.get("robust", {})
    if st.get("byT"):
        L += ["## 6. Робастность в базе (`robust.png`, `paths.png`)", "",
              "| T | mean | median | std | банкр. | доля сидов с PnL > 0 |", "|---|---|---|---|---|---|"]
        for T, e in st["byT"].items():
            pos = np.mean([t > 0 for t in e["totals"]])
            L.append(f"| {T} | {e['mean']:+.1f} | {e['median']:+.1f} | {e['std']:.1f} | {e['n_blown']}/{e['n']} | {pos*100:.0f}% |")
        L.append("")
    # итог
    L += ["## Итог", ""]
    concl = []
    base = R.get("base", {}).get("theta")
    if base:
        bd = on_boundary([base[n] for n in PARAM_NAMES])
        concl.append("оптимум интерьерный (не на границе диапазона)" if not bd else f"база на границе по {bd}")
    fl = R.get("flow", {})
    if "rows" in fl:
        rows = fl["rows"]; i = max(range(len(rows)), key=lambda k: rows[k]["mean"])
        if i + 1 < len(rows):
            concl.append(f"за пиком уступка хеджа растёт {rows[i]['conc2']*100:.2f}% → {rows[i+1]['conc2']*100:.2f}%: "
                         "ограничитель — глубина тонкой биржи")
    ec = R.get("econ", {}).get("variants", [])
    shares = [v["hedge_share_at_peak"] for v in ec if v["kind"] == "order_size"]
    if len(shares) >= 2:
        concl.append(f"доля приходящего объёма в пике по order_size: {min(shares)*100:.1f}%…{max(shares)*100:.1f}% "
                     + ("— инвариант, оптимум экономический" if max(shares) < 3 * min(shares) else "— не инвариант"))
    rb = R.get("robust", {}).get("byT", {})
    if rb:
        Tm = max(rb, key=int); e = rb[Tm]
        concl.append(f"в базе при T={Tm}: mean {e['mean']:+.1f}, банкротств {e['n_blown']}/{e['n']}")
    L += [f"- {c}" for c in concl] + [""]
    errs = [k for k, v in R.items() if isinstance(v, dict) and "error" in v]
    if errs:
        L += [f"Этапы с ошибками: {errs} (см. `log.txt`).", ""]
    with open(os.path.join(N.out, "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))


# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--deadline", default="07:00")
    a = ap.parse_args()
    hh, mm = (int(v) for v in a.deadline.split(":"))
    now = dt.datetime.now()
    deadline = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if deadline <= now:
        deadline += dt.timedelta(days=1)
    if a.smoke:
        deadline = now + dt.timedelta(hours=1)
    out = a.out or ("overnight_smoke" if a.smoke else "overnight")
    N = Night(out, deadline, a.smoke)
    N.stage("flow", lambda: stage_flow(N))
    N.stage("gp", lambda: stage_gp(N))
    N.stage("base", lambda: stage_base(N))
    N.stage("map1", lambda: stage_map1(N))
    N.stage("econ", lambda: stage_econ(N))
    N.stage("robust", lambda: stage_robust(N))
    N.stage("maps", lambda: stage_maps(N))
    N.save()
    N.log(f"готово, всего {(time.perf_counter() - N.t_start) / 60:.0f} мин")


if __name__ == "__main__":
    main()
