"""
flow_scan.py — стандартный тест модели: прибыль как функция скорости оборота.

Зачем. Из формулы контура (mean_field.md) все h_m входят в динамику только
через сопротивление R = sum 1/lambda_a, а ожидаемая прибыль растёт по потоку
1/R, пока что-то её не остановит. Поэтому любую правку модели (размер заявки,
идеальные арбитражёры, марки, риск, kappa, информированный фон) достаточно
проверить ОДНОЙ кривой: средний PnL против общего множителя s по всем h_m
(1/R ~ 1/s). Если кривая загибается внутри диапазона — есть интерьерный
оптимум, и диагностика говорит, ЧТО его создало:
    * растёт средняя уступка хеджа к миду  -> глубина книги (экономика);
    * появляются банкротства               -> капитал (экономика);
    * |ln P_X2| уходит от нуля             -> марки/шейдинг арбитражёров (артефакт);
    * ничего из этого, но прибыль растёт   -> граница: среда не ограничивает.

Учёт по умолчанию — исправленный (это то, что будет внесено в opt_simulation):
    * PnL и банкротство по справедливым маркам X1 = X2 = 1, Y_v = мид биржи v;
    * sigma^2 в риске арбитражёров — константа из волатильности бирж после
      прогрева (как gamma), а не EWMA собственного базиса на CE.
Старое поведение включается флагами (--ce-marks, --ewma-arb-vol) для сравнения.

Примеры:
    python flow_scan.py                                  # базовый тест
    python flow_scan.py --ideal-arbs                     # арбитражёры без шейдинга, lambda x1000
    python flow_scan.py --vary hr1 --scales 100,10,1,0.1 # риск склада T1
    python flow_scan.py --set kappa_t=20                 # любые поля SimConfig
    python flow_scan.py --base 1,1,1,1,1e-3,1e-3 --scales 10,3,1,0.3,0.1
Прогон 3000 тиков ~ 30 с; 8 точек x 6 сидов на 12 процессах ~ 3-4 мин.
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np
from joblib import Parallel, delayed

import agent_formulas as F
from agent_simulation import (EwmaVar, hedge_translators, submit_arb_orders,
                              submit_translator_orders)
from coupled_market import CoupledMarket
from opt_simulation import PARAM_NAMES, _NullRecorder, build_agents, opt_config
from pfx_exchange import Exchange as PortfolioExchange

ASSETS = ("X1", "Y1", "X2", "Y2")
AGENTS = ("T1", "T2", "AX", "AY")
VARY = {"hm": (0, 2, 4, 5), "hm_t": (0, 2), "hm_a": (4, 5), "hr1": (1,), "hr2": (3,), "hr": (1, 3)}


class FixedVar:
    """Замена EwmaVar: постоянная дисперсия."""
    def __init__(self, var: float):
        self.var = var
    def update(self, level: float) -> None:
        pass


def run_one(theta, seed: int, steps: int, fair_marks: bool, fixed_arb_vol: bool,
            overrides: dict) -> dict:
    cfg = opt_config(total_steps=steps, seed=seed, **overrides)
    market = CoupledMarket(cfg.venue1, cfg.venue2, tick_size=cfg.tick_size,
                           initial_price=cfg.initial_price,
                           anchor_ewma_half_life=cfg.anchor_half_life,
                           depth_band=cfg.depth_band, seed=cfg.seed,
                           fundamental_vol=cfg.fundamental_vol)
    market.warmup(cfg.warmup)
    ex1, ex2 = market.exchanges["1"], market.exchanges["2"]
    ce = PortfolioExchange(assets=list(F.ASSETS),
                           prices={"X1": 1.0, "Y1": ex1.mid, "X2": 1.0, "Y2": ex2.mid},
                           unit_of_account="X1")
    costs = [F.hedge_cost(e.spread, e.mid, cfg.tick_size) for e in (ex1, ex2)]
    costs = [c for c in costs if c is not None]
    sig2 = float(np.mean([e.volatility ** 2 for e in (ex1, ex2)]))
    gamma = (cfg.gamma if cfg.gamma is not None else F.calibrate_gamma(
        cost=float(np.mean(costs)) if costs else 0.5 * cfg.tick_size / cfg.initial_price,
        sigma2=sig2, q_max_fraction=cfg.q_max_fraction))
    agents = build_agents(theta)
    rec = _NullRecorder()
    if fixed_arb_vol:
        basis_vol = {k: FixedVar(sig2) for k in ("AX", "AY")}
    else:
        basis_vol = {k: EwmaVar(cfg.arb_vol_half_life, ce.rate(F.PORTFOLIOS[k]))
                     for k in ("AX", "AY")}

    # --- запись реальных хеджей: объём и уступка к миду -------------------- #
    hedged = {"1": 0.0, "2": 0.0}          # штук
    conc_sum = {"1": 0.0, "2": 0.0}        # sum(штук * относительная уступка)
    orig_exec = market.execute_hedges

    def exec_wrapped(venue, orders):
        mid = market.exchanges[venue].mid
        res = orig_exec(venue, orders)
        for name, r in res.items():
            if r["filled"] > 0 and r["avg_price"] is not None:
                c = (r["avg_price"] - mid) / mid
                if r["side"] == "sell":
                    c = -c
                hedged[venue] += r["filled"]
                conc_sum[venue] += r["filled"] * c
        return res
    market.execute_hedges = exec_wrapped

    c0 = cfg.c0
    flow = 0.0; n1sq = nAsq = 0.0; max_absSX2 = 0.0
    bankrupt, end = False, steps
    E_fair = E_ce = np.zeros(4)
    for t in range(1, steps + 1):
        market.step()
        submit_translator_orders(cfg, market, ce, agents, gamma)
        submit_arb_orders(cfg, ce, agents, gamma, basis_vol)
        rep = ce.step(dt=1.0)
        for f in rep.fills:
            if f.agent == "T1":
                flow += abs(f.notional)
        hedge_translators(cfg, market, ce, agents, gamma, rec, t)
        for k in ("AX", "AY"):
            basis_vol[k].update(ce.rate(F.PORTFOLIOS[k]))
        bal = np.array([[ce.balances.get(a.name, {}).get(x, 0.0) for x in ASSETS]
                        for a in agents])
        P_fair = np.array([1.0, ex1.mid, 1.0, ex2.mid])
        E_fair = bal @ P_fair
        E_ce = np.array([ce.mark_to_market(a.name) for a in agents])
        E = E_fair if fair_marks else E_ce
        n1sq += (bal[0, 1] * ex1.mid) ** 2
        nAsq += bal[2, 0] ** 2
        max_absSX2 = max(max_absSX2, abs(math.log(ce.price("X2"))))
        if np.abs(E).max() > c0:
            bankrupt, end = True, t
            break
    E = E_fair if fair_marks else E_ce
    return {"seed": seed, "total": float(E.sum()), "pnl": dict(zip(AGENTS, E.tolist())),
            "bankrupt": bankrupt, "end": end, "flow": flow / end,
            "hedge_units": {v: hedged[v] / end for v in hedged},
            "conc": {v: (conc_sum[v] / hedged[v] if hedged[v] > 0 else float("nan"))
                     for v in hedged},
            "rms_n1": math.sqrt(n1sq / end), "rms_nA": math.sqrt(nAsq / end),
            "max_absSX2": max_absSX2, "order_size": cfg.venue2.order_size}


def scan(base, idx, scales, seeds, steps, fair_marks, fixed_arb_vol, overrides, n_jobs):
    jobs = [(s, seed) for s in scales for seed in seeds]

    def job(s, seed):
        th = np.array(base, dtype=float)
        th[list(idx)] *= s
        r = run_one(tuple(th), seed, steps, fair_marks, fixed_arb_vol, overrides)
        r["s"] = s
        return r
    t0 = time.perf_counter()
    res = Parallel(n_jobs=min(n_jobs, len(jobs)))(delayed(job)(s, seed) for s, seed in jobs)
    rows = []
    for s in scales:
        rr = [r for r in res if r["s"] == s]
        c0 = 1000.0
        tot = [(-c0 if r["bankrupt"] else r["total"]) for r in rr]
        rows.append({
            "s": s, "mean": float(np.mean(tot)), "median": float(np.median(tot)),
            "std": float(np.std(tot)), "bankrupt": sum(r["bankrupt"] for r in rr), "n": len(rr),
            "flow": float(np.mean([r["flow"] for r in rr])),
            "levels2": float(np.mean([r["hedge_units"]["2"] for r in rr])) / rr[0]["order_size"],
            "conc1": float(np.nanmean([r["conc"]["1"] for r in rr])),
            "conc2": float(np.nanmean([r["conc"]["2"] for r in rr])),
            "rms_n1": float(np.mean([r["rms_n1"] for r in rr])),
            "rms_nA": float(np.mean([r["rms_nA"] for r in rr])),
            "absSX2": float(np.mean([r["max_absSX2"] for r in rr])),
            "pnl": {a: float(np.mean([r["pnl"][a] for r in rr])) for a in AGENTS},
        })
    return rows, time.perf_counter() - t0


def verdict(rows) -> str:
    best = max(range(len(rows)), key=lambda i: rows[i]["mean"])
    b = rows[best]
    if best == len(rows) - 1:
        return ("прибыль растёт до края диапазона — среда оборот не ограничивает; "
                "расширьте scales или меняйте среду")
    nxt = rows[best + 1]
    reasons = []
    if nxt["bankrupt"] > b["bankrupt"]:
        reasons.append(f"капитал: банкротств {b['bankrupt']} -> {nxt['bankrupt']}")
    if nxt["conc2"] > 1.3 * b["conc2"] or nxt["conc1"] > 1.3 * b["conc1"]:
        reasons.append(f"глубина книги: уступка хеджа {b['conc2']*100:.2f}% -> {nxt['conc2']*100:.2f}%")
    if nxt["absSX2"] > 0.2:
        reasons.append(f"артефакт марок/шейдинга арбитражёров: max|ln P_X2| = {nxt['absSX2']:.2f}")
    if not reasons:
        reasons.append("падение прибыли без явного диагностического признака — смотрите PnL по агентам")
    return f"оптимум при s = {b['s']:g} (mean {b['mean']:+.1f}); дальше рост остановлен: " + "; ".join(reasons)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="1,1,1,1,1,1", help="h_m1,h_r1,h_m2,h_r2,h_mx,h_my")
    ap.add_argument("--vary", default="hm", choices=sorted(VARY), help="какие параметры умножать на s")
    ap.add_argument("--scales", default="30,10,3,1,0.3,0.1,0.03,0.01")
    ap.add_argument("--seeds", type=int, default=6, help="число сидов (101, 102, ...)")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--ideal-arbs", action="store_true",
                    help="арбитражёры без шейдинга и с h_mx = h_my = 1e-3 (почти бесконечная lambda)")
    ap.add_argument("--ce-marks", action="store_true", help="старый учёт: PnL по ценам CE")
    ap.add_argument("--ewma-arb-vol", action="store_true", help="старый риск арбитражёров: EWMA базиса CE")
    ap.add_argument("--set", action="append", default=[], metavar="FIELD=VALUE",
                    help="поле SimConfig, например kappa_t=20 или arb_shading=False")
    a = ap.parse_args()

    base = [float(x) for x in a.base.split(",")]
    overrides = {}
    for kv in a.set:
        k, v = kv.split("=", 1)
        overrides[k] = {"True": True, "False": False}.get(v, None)
        if overrides[k] is None:
            overrides[k] = float(v) if "." in v or "e" in v.lower() else int(v)
    if a.ideal_arbs:
        overrides["arb_shading"] = False
        base[4] = base[5] = 1e-3
    scales = [float(x) for x in a.scales.split(",")]
    seeds = [101 + i for i in range(a.seeds)]
    idx = VARY[a.vary]

    print(f"база {dict(zip(PARAM_NAMES, base))}; умножаем {[PARAM_NAMES[i] for i in idx]} на s; "
          f"{a.steps} тиков, сиды {seeds}; марки {'CE' if a.ce_marks else 'справедливые'}, "
          f"риск арбитражёров {'EWMA базиса' if a.ewma_arb_vol else 'константа'}; overrides {overrides}")
    rows, el = scan(base, idx, scales, seeds, a.steps, not a.ce_marks, not a.ewma_arb_vol,
                    overrides, a.jobs)
    print(f"({el:.0f} с)\n{'s':>6} | {'mean':>8} {'median':>8} {'std':>7} {'банкр':>5} | {'поток X1/тик':>12} "
          f"{'уровней2/тик':>12} {'уступка1':>8} {'уступка2':>8} | {'rms n1':>7} {'rms nA':>7} {'max|lnPX2|':>10}")
    for r in rows:
        print(f"{r['s']:>6g} | {r['mean']:>+8.1f} {r['median']:>+8.1f} {r['std']:>7.1f} {r['bankrupt']:>3}/{r['n']:<1} | "
              f"{r['flow']:>12.3f} {r['levels2']:>12.3f} {r['conc1']*100:>7.2f}% {r['conc2']*100:>7.2f}% | "
              f"{r['rms_n1']:>7.1f} {r['rms_nA']:>7.1f} {r['absSX2']:>10.3f}")
        print(f"{'':>6}   PnL по агентам: " + "  ".join(f"{k}={v:+.1f}" for k, v in r["pnl"].items()))
    print("\nВердикт:", verdict(rows))


if __name__ == "__main__":
    main()
