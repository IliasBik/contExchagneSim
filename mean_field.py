"""
mean_field.py — аналитическое (mean-field) приближение задачи optimize_agents.

Идея: при четырёх заявках клиринг CE решается в явном виде — это «контур»
из четырёх последовательных проводимостей lambda_a с источником напряжения
delta = ln(m1/m2) и «конденсаторами» (шейдинг g_a * n_a):

    eps = delta - g1 n1 - g2 n2 - (gX + gY) nA,   V = eps / R,
    R = 1/lam1 + 1/lam2 + 1/lamX + 1/lamY,        n_a <- n_a + V.

Лимитные биржи заменяются экзогенным шумом мидов eta_v = ln(m_v / anchor)
(VAR(1), калибруется по прогону рынка без агентов) и параметрической
стоимостью хеджа: предельная относительная потеря c_v + k_v * x на глубине x
штук. Хедж транслятора — правило hedge_plan для линейной книги:
    h = clip((g |u| - c) / (g m + k), 0, |q|).

Три уровня модели, от точного к аналитическому:
    reduced_run  — тот же цикл, что в opt_simulation, но с явным клирингом и
                   экзогенным рынком (записанные пути рынка без агентов или
                   синтетические VAR(1)); PnL каждого агента считается точно.
    lg_eval      — линейно-гауссовская модель: ковариация состояния
                   (n1, n2, nA, eta1, eta2) на горизонте T со старта из нуля,
                   мёртвая зона хеджа — статистической линеаризацией; даёт
                   E[PnL] и вероятность взрыва в замкнутой форме (миллисекунды).
    lg_optimize  — максимизация E[PnL] по 6 параметрам при ограничении на
                   вероятность взрыва — секунды вместо часов.

Запуск:  python mean_field.py [full_runs.json]
         (калибровка кэшируется в mean_field_calib.json / _paths.npz;
          необязательный аргумент — результаты полной симуляции для сравнения)
"""

from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np
from joblib import Parallel, delayed
from scipy.linalg import solve_discrete_lyapunov
from scipy.optimize import minimize
from scipy.stats import norm

import agent_formulas as F
from agent_simulation import SimConfig
from coupled_market import CoupledMarket

PARAM_NAMES = ("h_m1", "h_r1", "h_m2", "h_r2", "h_mx", "h_my")
AGENTS = ("T1", "T2", "AX", "AY")
CALIB_FILE = "mean_field_calib.json"
PATHS_FILE = "mean_field_paths.npz"
RESULT_FILE = "mean_field_result.json"
LOG_BOUNDS = (-4.0, math.log10(300.0))      # как в optimize_agents
T_RUN = 6000
BLOWUP = 1000.0


# --------------------------------------------------------------------------- #
# 1. Калибровка экзогенного рынка (прогон CoupledMarket без агентов)
# --------------------------------------------------------------------------- #

QUOTE_SIZES = (0.05, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0)


def _market_stats(seed: int, steps: int) -> dict:
    cfg = SimConfig(seed=seed)
    mk = CoupledMarket(cfg.venue1, cfg.venue2, tick_size=cfg.tick_size,
                       initial_price=cfg.initial_price,
                       anchor_ewma_half_life=cfg.anchor_half_life,
                       depth_band=cfg.depth_band, seed=seed,
                       fundamental_vol=cfg.fundamental_vol)
    mk.warmup(cfg.warmup)
    ex = [mk.exchanges["1"], mk.exchanges["2"]]
    # gamma калибруется симуляцией по состоянию после прогрева
    costs = [F.hedge_cost(e.spread, e.mid, cfg.tick_size) for e in ex]
    costs = [c for c in costs if c is not None]
    gamma = F.calibrate_gamma(cost=float(np.mean(costs)),
                              sigma2=float(np.mean([e.volatility ** 2 for e in ex])),
                              q_max_fraction=cfg.q_max_fraction)
    mid = np.zeros((steps, 2)); anchor = np.zeros(steps)
    spread = np.full((steps, 2), np.nan)
    vol2 = np.zeros((steps, 2)); H = np.zeros((steps, 2))
    loss = []                          # (venue, side, size) относительная потеря
    for t in range(steps):
        mk.step()
        anchor[t] = mk.anchor
        for v, e in enumerate(ex):
            mid[t, v] = e.mid
            if e.spread is not None:
                spread[t, v] = e.spread
            vol2[t, v] = e.volatility ** 2
            H[t, v] = F.book_quality(e.depth_near_mid(cfg.depth_band), e.mid,
                                     e.spread, cfg.tick_size)
        if t % 20 == 0:
            row = np.full((2, 2, len(QUOTE_SIZES)), np.nan)
            for v, e in enumerate(ex):
                for si, side in enumerate(("sell", "buy")):
                    for j, s in enumerate(QUOTE_SIZES):
                        q = e.quote(side, s)
                        if q["avg_price"] is not None and q["filled"] >= s - 1e-9:
                            row[v, si, j] = ((e.mid - q["avg_price"]) if side == "sell"
                                             else (q["avg_price"] - e.mid)) / e.mid
            loss.append(row)
    return {"mid": mid, "anchor": anchor, "spread": spread, "vol2": vol2, "H": H,
            "loss": np.array(loss), "gamma": gamma}


def calibrate(seeds=(101, 102, 103, 104, 105, 106), steps: int = T_RUN,
              force: bool = False, n_jobs: int = 6) -> dict:
    """Статистика рынка без агентов; кэшируется в CALIB_FILE + PATHS_FILE."""
    if not force and os.path.exists(CALIB_FILE) and os.path.exists(PATHS_FILE):
        with open(CALIB_FILE, encoding="utf-8") as f:
            st = json.load(f)
        for k in ("Phi", "Q"):
            st[k] = np.array(st[k])
        return st

    t0 = time.perf_counter()
    runs = Parallel(n_jobs=min(n_jobs, len(seeds)))(
        delayed(_market_stats)(s, steps) for s in seeds)
    eta = np.concatenate([np.log(r["mid"] / r["anchor"][:, None]) for r in runs])
    dlna = np.concatenate([np.diff(np.log(r["anchor"])) for r in runs])
    x, y = eta[:-1], eta[1:]
    Phi = np.linalg.lstsq(x, y, rcond=None)[0].T          # eta_{t+1} = Phi eta_t + xi
    Q = np.cov((y - x @ Phi.T).T)
    loss = np.concatenate([r["loss"] for r in runs])       # (snap, venue, side, size)
    sizes = np.array(QUOTE_SIZES)
    c, k = [], []
    for v in range(2):
        mean_loss = np.nanmean(loss[:, v, :, :], axis=(0, 1))      # по размерам
        c.append(float(mean_loss[sizes <= 0.5].mean()))
        slope = np.polyfit(sizes[sizes >= 1.0], mean_loss[sizes >= 1.0], 1)[0]
        k.append(float(max(2.0 * slope, 0.0)))     # средняя потеря -> предельная
    cfg = SimConfig()
    st = {
        "m0": cfg.initial_price,
        "c": c, "k": k,
        "H": [float(np.mean([r["H"][:, v].mean() for r in runs])) for v in range(2)],
        "p0": float(np.mean([(r["H"] == 0).any(axis=1).mean() for r in runs])),
        "sig2": [float(np.mean([r["vol2"][:, v].mean() for r in runs])) for v in range(2)],
        "gamma": float(np.mean([r["gamma"] for r in runs])),
        "Phi": Phi.tolist(), "Q": Q.tolist(),
        "eta_std": eta.std(axis=0).tolist(),
        "sig_a": float(dlna.std()),
        "c0": cfg.c0, "kappa_t": cfg.kappa_t, "kappa_a": cfg.kappa_a,
        "floor": cfg.capital_floor_frac * cfg.c0, "clamp": cfg.shading_clamp,
        "basis_hl": cfg.arb_vol_half_life,
        "seeds": list(seeds), "steps": steps,
        "elapsed_s": time.perf_counter() - t0,
    }
    with open(CALIB_FILE, "w", encoding="utf-8") as f:
        json.dump(st, f, indent=1, ensure_ascii=False)
    # записанные пути для редуцированной симуляции (индекс 0 = состояние старта)
    def pre(a):
        return np.concatenate([a[:1], a], axis=0)
    np.savez(PATHS_FILE, seeds=np.array(seeds),
             eta=np.array([pre(np.log(r["mid"] / r["anchor"][:, None])) for r in runs]),
             lna=np.array([pre(np.log(r["anchor"])) for r in runs]),
             c=np.array([pre(r["spread"] / (2.0 * r["mid"])) for r in runs]),
             H=np.array([pre(r["H"]) for r in runs]),
             sig2=np.array([pre(r["vol2"]) for r in runs]),
             gamma=np.array([r["gamma"] for r in runs]))
    st["Phi"], st["Q"] = Phi, Q
    return st


def recorded_paths() -> list[dict]:
    d = np.load(PATHS_FILE)
    out = []
    for i in range(len(d["seeds"])):
        H = d["H"][i]
        out.append({"eta": d["eta"][i], "lna": d["lna"][i], "c": d["c"][i], "H": H,
                    "sig2": d["sig2"][i], "trade": (H > 0).all(axis=1),
                    "gamma": float(d["gamma"][i]), "seed": int(d["seeds"][i])})
    return out


def print_calibration(st: dict) -> None:
    print("Калибровка экзогенного рынка (без агентов):")
    print(f"  полуспред c = {st['c'][0]*100:.2f}% / {st['c'][1]*100:.2f}%   "
          f"предельный импакт k = {st['k'][0]*100:.2f}% / {st['k'][1]*100:.2f}% на штуку")
    print(f"  H = {st['H'][0]:.1f} / {st['H'][1]:.1f}   sigma^2 = {st['sig2'][0]:.2e} / "
          f"{st['sig2'][1]:.2e}   gamma = {st['gamma']:.4g}   p0(нет T2) = {st['p0']:.3f}")
    Phi, Q = st["Phi"], st["Q"]
    print(f"  eta = ln(mid/anchor): std = {st['eta_std'][0]*100:.2f}% / {st['eta_std'][1]*100:.2f}%,"
          f"  VAR(1): diag(Phi) = {Phi[0,0]:.3f} / {Phi[1,1]:.3f}, "
          f"std(xi) = {math.sqrt(Q[0,0])*100:.2f}% / {math.sqrt(Q[1,1])*100:.2f}%,"
          f"  std(d ln anchor) = {st['sig_a']*100:.3f}%")


# --------------------------------------------------------------------------- #
# 2. Эффективные параметры
# --------------------------------------------------------------------------- #

def effective(theta, st: dict) -> dict:
    h_m1, h_r1, h_m2, h_r2, h_mx, h_my = (float(v) for v in theta)
    c0, m0 = st["c0"], st["m0"]
    lam = np.array([st["kappa_t"] * st["H"][0] / h_m1,
                    st["kappa_t"] * st["H"][1] / h_m2,
                    st["kappa_a"] * c0 / h_mx,
                    st["kappa_a"] * c0 / h_my])
    R = float(np.sum(1.0 / lam)) / (1.0 - st["p0"])     # p0 — тики без сделок
    g = np.array([h_r1 * st["gamma"] * st["sig2"][0] / c0,
                  h_r2 * st["gamma"] * st["sig2"][1] / c0])
    phi = g * m0 / (g * m0 + np.array(st["k"]))          # доля излишка, хеджируемая за тик
    nstar = np.array(st["c"]) / g                         # мёртвая зона, X1
    return {"lam": lam, "R": R, "g": g, "phi": phi, "nstar": nstar}


# --------------------------------------------------------------------------- #
# 3. Редуцированная симуляция (точный клиринг, экзогенный рынок)
# --------------------------------------------------------------------------- #

def synth_paths(st: dict, T: int, n_paths: int, seed: int = 0) -> list[dict]:
    """Синтетические пути: VAR(1) для eta, случайное блуждание якоря,
    бернуллиевские тики без сделок; c, H, sigma^2 — константы."""
    rng = np.random.default_rng(seed)
    Phi, Q = st["Phi"], st["Q"]
    L = np.linalg.cholesky(Q)
    Ls = np.linalg.cholesky(solve_discrete_lyapunov(Phi, Q))   # стационарный старт
    out = []
    for _ in range(n_paths):
        eta = np.zeros((T + 1, 2))
        eta[0] = Ls @ rng.standard_normal(2)
        xi = rng.standard_normal((T, 2)) @ L.T
        for t in range(T):
            eta[t + 1] = Phi @ eta[t] + xi[t]
        lna = math.log(st["m0"]) + np.concatenate(
            [[0.0], np.cumsum(st["sig_a"] * rng.standard_normal(T))])
        out.append({"eta": eta, "lna": lna, "trade": rng.random(T + 1) >= st["p0"]})
    return out


def reduced_run(theta, st: dict, path: dict, blowup_limit: float | None = BLOWUP,
                record: bool = False) -> dict:
    """Один прогон редуцированной модели. Возвращает PnL агентов, total,
    blown, RMS инвентарей и обороты."""
    h_m1, h_r1, h_m2, h_r2, h_mx, h_my = (float(v) for v in theta)
    c0, floor, clamp = st["c0"], st["floor"], st["clamp"]
    kt, ka = st["kappa_t"], st["kappa_a"]
    gamma = path.get("gamma", st["gamma"])
    k1, k2 = st["k"]
    alpha = 1.0 - 0.5 ** (1.0 / st["basis_hl"])
    eta, lna, trade = path["eta"], path["lna"], path["trade"]
    T = len(eta) - 1
    c_t = path.get("c"); H_t = path.get("H"); s2_t = path.get("sig2")
    c1 = c2 = H1 = H2 = s21 = s22 = 0.0
    if c_t is None:
        c1, c2 = st["c"]
    if H_t is None:
        H1, H2 = st["H"]
    if s2_t is None:
        s21, s22 = st["sig2"]

    Q = np.zeros((4, 4))                       # агенты x активы (X1,Y1,X2,Y2)
    S = np.array([0.0, lna[0] + eta[0, 0], 0.0, lna[0] + eta[0, 1]])
    varX = varY = 0.0
    prev_rX, prev_rY = 0.0, S[1] - S[3]
    blown, end = False, T
    sum_n1sq = sum_nAsq = sum_absV = turn1 = turn2 = 0.0
    E = np.zeros(4)
    if record:
        eq = np.zeros((4, T + 1))

    def clip(x):
        return min(max(x, -clamp), clamp)

    for t in range(1, T + 1):
        if c_t is not None:
            c1, c2 = c_t[t]
        if H_t is not None:
            H1, H2 = H_t[t]
        if s2_t is not None:
            s21, s22 = s2_t[t]
        m1 = math.exp(lna[t] + eta[t, 0]); m2 = math.exp(lna[t] + eta[t, 1])
        P = np.exp(S)
        E = Q @ P
        C = c0 + E
        lam1 = kt * H1 / h_m1 * (1.0 if C[0] > 0 else 0.0)
        lam2 = kt * H2 / h_m2 * P[2] * (1.0 if C[1] > 0 else 0.0)
        lamX = ka * max(C[2], 0.0) / h_mx
        lamY = ka * max(C[3], 0.0) / h_my
        if trade[t] and min(lam1, lam2, lamX, lamY) > 0.0:
            live = 4
        elif lam2 <= 0.0 and min(lam1, lamX, lamY) > 0.0:
            live = 3                       # нет T2: три заявки = дерево, цены без сделок
        else:
            live = 0
        if live == 3:
            g1 = h_r1 * gamma * s21 / max(C[0], floor)
            gX = gamma * varX / max(C[2], floor)
            gY = gamma * varY / max(C[3], floor)
            lz1 = math.log(m1) - clip(g1 * Q[0, 1] * m1)
            lzX = -clip(gX * Q[2, 0] * P[0])
            lzY = -clip(gY * Q[3, 1] * P[1])
            S = np.array([0.0, lz1, -lzX, lz1 - lzY])
            P = np.exp(S)
        if live == 4:
            g1 = h_r1 * gamma * s21 / max(C[0], floor)
            g2 = h_r2 * gamma * s22 / max(C[1], floor)
            gX = gamma * varX / max(C[2], floor)
            gY = gamma * varY / max(C[3], floor)
            lz1 = math.log(m1) - clip(g1 * Q[0, 1] * m1)
            lz2 = math.log(m2) - clip(g2 * Q[1, 3] * m2)
            lzX = -clip(gX * Q[2, 0] * P[0])
            lzY = -clip(gY * Q[3, 1] * P[1])
            eps = lz1 - lz2 + lzX - lzY
            R = 1.0 / lam1 + 1.0 / lam2 + 1.0 / lamX + 1.0 / lamY
            V = eps / R
            SY1 = lz1 - eps / (R * lam1)
            SX2 = -lzX + eps / (R * lamX)
            SY2 = SX2 + lz2 + eps / (R * lam2)
            S = np.array([0.0, SY1, SX2, SY2])
            P = np.exp(S)
            # филлы: T1 +V(Y1,-X1), T2 -V(Y2,-X2), AX +V(X1,-X2), AY -V(Y1,-Y2)
            Q[0, 1] += V / P[1]; Q[0, 0] -= V
            Q[1, 3] -= V / P[3]; Q[1, 2] += V / P[2]
            Q[2, 0] += V;        Q[2, 2] -= V / P[2]
            Q[3, 1] -= V / P[1]; Q[3, 3] += V / P[3]
            sum_absV += abs(V)
        # хеджи трансляторов об линейную книгу
        E = Q @ P
        C = c0 + E
        for ai, yi, xi, m, cc, kk, hr, s2 in ((0, 1, 0, m1, c1, k1, h_r1, s21),
                                               (1, 3, 2, m2, c2, k2, h_r2, s22)):
            q = Q[ai, yi]
            u = q * m
            if abs(u) < 1e-9 or not (cc == cc):      # нет позиции / пустая сторона
                continue
            g = hr * gamma * s2 / max(C[ai], floor)
            h = (g * abs(u) - cc) / (g * m + kk)
            h = min(max(h, 0.0), abs(q))
            if h <= 0.0:
                continue
            if u > 0:      # продажа Y об биды
                Q[ai, yi] -= h
                Q[ai, xi] += h * m * (1.0 - cc) - kk * h * h * m / 2.0
            else:          # откуп Y об аски
                Q[ai, yi] += h
                Q[ai, xi] -= h * m * (1.0 + cc) + kk * h * h * m / 2.0
            if ai == 0:
                turn1 += h * m
            else:
                turn2 += h * m
        rX, rY = -S[2], S[1] - S[3]
        varX = (1 - alpha) * varX + alpha * (rX - prev_rX) ** 2
        varY = (1 - alpha) * varY + alpha * (rY - prev_rY) ** 2
        prev_rX, prev_rY = rX, rY
        E = Q @ P
        sum_n1sq += (Q[0, 1] * m1) ** 2
        sum_nAsq += (Q[2, 0] * P[0]) ** 2
        if record:
            eq[:, t] = E
        if blowup_limit is not None and np.abs(E).max() > blowup_limit:
            blown, end = True, t
            break
    P_fair = np.array([1.0, m1, 1.0, m2])
    out = {"pnl": {a: float(E[i]) for i, a in enumerate(AGENTS)},
           "total": float(E.sum()), "total_fair": float((Q @ P_fair).sum()),
           "P_ce": P.copy(), "blown": blown, "end_tick": end,
           "rms_n1": math.sqrt(sum_n1sq / end), "rms_nA": math.sqrt(sum_nAsq / end),
           "flow": sum_absV / end, "turn1": turn1 / end, "turn2": turn2 / end}
    if record:
        out["equity"] = eq
    return out


def reduced_eval(theta, st: dict, paths: list[dict], penalty: float = BLOWUP) -> dict:
    """Медиана/среднее total по путям — тот же протокол, что в optimize_agents."""
    runs = [reduced_run(theta, st, p) for p in paths]
    totals = [-penalty if r["blown"] else r["total"] for r in runs]
    ok = [r for r in runs if not r["blown"]]
    return {"median": float(np.median(totals)), "mean": float(np.mean(totals)),
            "std": float(np.std(totals)), "n_blown": len(runs) - len(ok),
            "n": len(runs),
            "median_fair": float(np.median([-penalty if r["blown"] else r["total_fair"] for r in runs])),
            "P_X2": float(np.mean([r["P_ce"][2] for r in runs])),
            "pnl": {a: (float(np.mean([r["pnl"][a] for r in ok])) if ok else 0.0)
                    for a in AGENTS},
            "rms_n1": float(np.mean([r["rms_n1"] for r in runs])),
            "rms_nA": float(np.mean([r["rms_nA"] for r in runs])),
            "flow": float(np.mean([r["flow"] for r in runs])),
            "turn1": float(np.mean([r["turn1"] for r in runs])),
            "turn2": float(np.mean([r["turn2"] for r in runs]))}


# --------------------------------------------------------------------------- #
# 4. Линейно-гауссовская модель на конечном горизонте
# --------------------------------------------------------------------------- #

def _horizon_cov(A: np.ndarray, W: np.ndarray, S0: np.ndarray, T: int):
    """x_t = A x_{t-1} + w_t, Cov w = W, Cov x_0 = S0.
    Возвращает (Cov x_T, среднее по t=1..T от Cov x_t) — удвоением, O(log T)."""
    def combine(a, b):
        # блок = (P, U, V, Y, Apow, n):  P(n) = sum_{k<n} A^k W A^k^T,
        # U(n) = sum_{t<=n} P(t),  V(n) = sum_{t<=n} A^t S0 A^t^T,  Y(n) = A^n S0 A^n^T
        Pa, Ua, Va, Ya, Aa, na = a
        Pb, Ub, Vb, Yb, Ab, nb = b
        P = Pa + Aa @ Pb @ Aa.T
        U = Ua + nb * Pa + Aa @ Ub @ Aa.T
        V = Va + Aa @ Vb @ Aa.T
        Y = Aa @ Yb @ Aa.T
        return P, U, V, Y, Aa @ Ab, na + nb
    Y1 = A @ S0 @ A.T
    blocks = [(W.copy(), W.copy(), Y1, Y1, A.copy(), 1)]
    while blocks[-1][5] * 2 <= T:
        blocks.append(combine(blocks[-1], blocks[-1]))
    acc, rem = None, T
    for blk in reversed(blocks):
        if blk[5] <= rem:
            acc = blk if acc is None else combine(acc, blk)
            rem -= blk[5]
    P, U, V, Y, _, n = acc
    return P + Y, (U + V) / n


def _deadzone_moments(phi: float, nstar: float, sigma: float) -> tuple[float, float, float]:
    """h = phi * (|n| - n*)_+ sign(n), n ~ N(0, sigma^2):
    возвращает (beta_eq, E|h|, E h^2), beta_eq = E[h n] / E[n^2]."""
    if sigma <= 0.0:
        return 0.0, 0.0, 0.0
    kap = nstar / sigma
    tail = norm.sf(kap); dens = norm.pdf(kap)
    beta = 2.0 * phi * tail
    e_abs = 2.0 * phi * sigma * (dens - kap * tail)
    e_sq = 2.0 * phi * phi * sigma * sigma * ((1.0 + kap * kap) * tail - kap * dens)
    return beta, e_abs, e_sq


def lg_eval(theta, st: dict, T: int = T_RUN, n_iter: int = 80,
            p_blow_B: float = BLOWUP) -> dict:
    """Моменты линейно-гауссовской модели на горизонте T.

    Состояние x_t = (n1'', n2'', nA'', eta1, eta2): инвентари после хеджей
    (в стоимости, X1) и шум мидов. За тик:
        n'  = (I - 1 g^T / R) n'' + (1/R) 1 d^T eta,     d = (1, -1)
        h_v = beta_v n'_v,   n''_v = (1 - beta_v) n'_v   (v = 1, 2; nA без утечки)
        eta_{t+1} = Phi eta_t + xi.
    E[PnL] за тик:
        pi = E[h1 eta1] - E[h2 eta2] - sum_v (c_v E|h_v| + k_v E[h_v^2] / (2 m)).
    beta_v и моменты |h| — статистическая линеаризация мёртвой зоны (точна для
    гауссовского n'); g арбитражёров — самосогласованно через дисперсию
    приращений базиса на CE. Вероятность взрыва: PnL агента i за тик меняется
    на (экспозиция_i) x (приращение его марки); P(max|PnL_i| > B) ~
    4 Phi_bar(B / sqrt(T E[e_i^2] Var(d mark_i))).
    """
    ef = effective(theta, st)
    R, lam = ef["R"], ef["lam"]
    m0, c0, gamma = st["m0"], st["c0"], st["gamma"]
    c, k = np.array(st["c"]), np.array(st["k"])
    Phi, Qx = st["Phi"], st["Q"]
    one = np.ones(3); d = np.array([1.0, -1.0])
    e1, e2, eA = np.eye(3)
    Sig_eta = solve_discrete_lyapunov(Phi, Qx)
    S0 = np.zeros((5, 5)); S0[3:, 3:] = Sig_eta          # старт: инвентари 0
    var_ddelta = float(d @ (2.0 * Sig_eta - Phi @ Sig_eta - Sig_eta @ Phi.T) @ d)
    gA = np.array([gamma * var_ddelta / (R * lam[2]) ** 2 / c0,
                   gamma * var_ddelta / (R * lam[3]) ** 2 / c0])
    beta = ef["phi"].copy()
    bad = {"pi": float("nan"), "p_blow": 1.0, "harvest": np.full(2, np.nan),
           "cost": np.full(2, np.nan), "beta": beta, "rms_n": np.full(3, np.inf),
           "rms_n_T": np.full(3, np.inf), "gA": gA, "R": R, "lam": lam, "g": ef["g"],
           "nstar": ef["nstar"], "phi": ef["phi"], "e_abs_h": np.full(2, np.nan),
           "risk": np.full(4, np.inf), "sig_nprime": np.full(3, np.inf),
           "runaway": np.inf, "discharge": np.inf, "flow_rms": np.inf}

    def build(beta, gA):
        g = np.array([ef["g"][0], ef["g"][1], gA.sum()])
        M = np.eye(3) - np.outer(one, g) / R
        IB = np.diag([1.0 - beta[0], 1.0 - beta[1], 1.0])
        A = np.zeros((5, 5)); G = np.zeros((5, 2))
        A[:3, :3] = IB @ M
        A[:3, 3:] = IB @ np.outer(one, d) @ Phi / R
        A[3:, 3:] = Phi
        G[:3, :] = IB @ np.outer(one, d) / R
        G[3:, :] = np.eye(2)
        return g, M, A, G

    for _ in range(n_iter):
        g, M, A, G = build(beta, gA)
        if np.abs(np.linalg.eigvals(A)).max() > 1.0 + 1e-9:
            return bad                     # контур неустойчив (перезарядка за тик)
        Sig_T, Sig = _horizon_cov(A, G @ Qx @ G.T, S0, T)
        if not np.all(np.isfinite(Sig)):
            return bad
        # n'_t = Cn x_{t-1} + Dn xi_t;   eta_t = Ce x_{t-1} + xi_t
        Cn = np.hstack([M, np.outer(one, d) @ Phi / R]); Dn = np.outer(one, d) / R
        Ce = np.hstack([np.zeros((2, 3)), Phi])
        cov_n_eta = Cn @ Sig @ Ce.T + Dn @ Qx            # 3x2
        var_n = np.diag(Cn @ Sig @ Cn.T + Dn @ Qx @ Dn.T)
        sig_n = np.sqrt(np.maximum(var_n, 0.0))
        new_beta = np.zeros(2); e_abs = np.zeros(2); e_sq = np.zeros(2)
        for v in range(2):
            new_beta[v], e_abs[v], e_sq[v] = _deadzone_moments(
                ef["phi"][v], ef["nstar"][v], sig_n[v])

        # дисперсия приращений линейного выхода y_t = cy x_{t-1} + ey xi_t
        def incr_var(cy, ey):
            var_y = cy @ Sig @ cy + ey @ Qx @ ey
            cov_lag = cy @ A @ Sig @ cy + cy @ G @ Qx @ ey
            return max(2.0 * var_y - 2.0 * cov_lag, 0.0)

        def mark(coef_n, coef_eta, w_eps):
            # y = coef_n . n'' + coef_eta . eta_t + w_eps * eps_t,
            # eps_t = d^T eta_t - g^T n''_{t-1},  eta_t = Phi eta_{t-1} + xi_t
            cy = np.zeros(5); ey = np.zeros(2)
            cy[:3] = coef_n - w_eps * g
            cy[3:] = (coef_eta + w_eps * d) @ Phi
            ey[:] = coef_eta + w_eps * d
            return incr_var(cy, ey)

        v_AX = mark(-gA[0] * eA, np.zeros(2), -1.0 / (R * lam[2]))
        v_AY = mark(+gA[1] * eA, np.zeros(2), +1.0 / (R * lam[3]))
        gA_new = gamma * np.array([v_AX, v_AY]) / c0
        done = (np.allclose(new_beta, beta, rtol=1e-6, atol=1e-12)
                and np.allclose(gA_new, gA, rtol=1e-6, atol=1e-15))
        beta = 0.5 * beta + 0.5 * new_beta
        gA = 0.5 * gA + 0.5 * gA_new
        if done:
            break

    harvest = np.array([beta[0] * cov_n_eta[0, 0], -beta[1] * cov_n_eta[1, 1]])
    cost = c * e_abs + k * e_sq / (2.0 * m0)
    pi = float(harvest.sum() - cost.sum())
    # --- риск взрыва по агентам ------------------------------------------- #
    ex1 = np.array([1.0, 0.0, 0.0]); exA = np.array([0.0, 0.0, 1.0])
    ex2 = np.array([0.0, -2.0, 1.0])                 # T2: CE-шорт + купленное на бирже
    v_T1 = mark(-ef["g"][0] * e1, np.array([1.0, 0.0]), -1.0 / (R * lam[0])) + st["sig_a"] ** 2
    w2 = (1.0 / lam[2] + 1.0 / lam[1]) / R
    v_T2 = mark(ef["g"][1] * e2 + gA[0] * eA, np.array([0.0, 1.0]), w2) + st["sig_a"] ** 2
    Sn = Sig[:3, :3]
    e2_T = np.array([ex1 @ Sn @ ex1, ex2 @ Sn @ ex2, exA @ Sn @ exA, exA @ Sn @ exA])
    v_mark = np.array([v_T1, v_T2, v_AX, v_AY])
    risk = np.sqrt(T * e2_T * v_mark)                # std PnL_i(T)
    p_blow = float(min(1.0, np.sum(4.0 * norm.sf(p_blow_B / np.maximum(risk, 1e-12)))))
    # --- срыв шейдинга арбитражёров -------------------------------------- #
    # g_A = gamma Var(d basis)/c0, Var(d basis) ~ g_A^2 Var(V) + phi_A^2 Var(d eps):
    # уравнение g = a g^2 + b имеет корни только при 4ab <= 1; при 4ab -> 1
    # малый (устойчивый) корень сливается с большим, шейдинг уходит в клэмп.
    cy_e = np.concatenate([-g, d @ Phi]); ey_e = d.copy()
    var_eps = float(cy_e @ Sig @ cy_e + ey_e @ Qx @ ey_e)
    var_deps = incr_var(cy_e, ey_e)
    a_run = gamma * var_eps / R ** 2 / c0
    b_run = gamma * var_deps / c0 * np.array([1.0 / (R * lam[2]) ** 2,
                                              1.0 / (R * lam[3]) ** 2])
    runaway = float(np.max(4.0 * a_run * b_run))
    discharge = float((ef["g"][0] + ef["g"][1] * (1.0 - beta[1]) + gA.sum()) / R)
    return {"pi": pi, "p_blow": p_blow, "risk": risk, "harvest": harvest, "cost": cost,
            "runaway": runaway, "discharge": discharge,
            "flow_rms": math.sqrt(var_eps) / R,
            "beta": beta, "sig_nprime": sig_n, "rms_n": np.sqrt(np.maximum(np.diag(Sn), 0)),
            "rms_n_T": np.sqrt(np.maximum(np.diag(Sig_T[:3, :3]), 0)),
            "gA": gA, "R": R, "lam": lam, "g": ef["g"], "nstar": ef["nstar"],
            "phi": ef["phi"], "e_abs_h": e_abs}


# --------------------------------------------------------------------------- #
# 5. Оптимизация аналитической модели
# --------------------------------------------------------------------------- #

RUNAWAY_MAX = 0.5      # 4ab <= 0.5: запас до слияния корней (срыв шейдинга арбитражёров)
DISCHARGE_MAX = 1.0    # разряд контура за тик < 1: без перезарядки и осцилляций


def lg_objective(u, st: dict, T: int, p_max: float, penalty: float = 2e4) -> float:
    theta = 10.0 ** np.asarray(u)
    r = lg_eval(theta, st, T)
    if not np.isfinite(r["pi"]):
        return 1e6
    pen = (max(0.0, r["p_blow"] - p_max) ** 2
           + max(0.0, r["runaway"] / RUNAWAY_MAX - 1.0) ** 2
           + max(0.0, r["discharge"] / DISCHARGE_MAX - 1.0) ** 2)
    return -(T * r["pi"]) + penalty * pen


def lg_optimize(st: dict, T: int = T_RUN, p_max: float = 0.25, starts=None,
                seed: int = 0, n_random: int = 4) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    lo, hi = LOG_BOUNDS
    if starts is None:
        starts = ([np.zeros(6), np.log10([0.263, 0.0646, 0.02636, 4.301, 0.01115, 0.03708])]
                  + [rng.uniform(lo, hi, 6) for _ in range(n_random)])
    best_u, best_f = None, np.inf
    for u0 in starts:
        res = minimize(lg_objective, u0, args=(st, T, p_max), method="Nelder-Mead",
                       options={"maxiter": 500, "xatol": 1e-3, "fatol": 1e-4})
        u = np.clip(res.x, lo, hi)
        f = lg_objective(u, st, T, p_max)
        if f < best_f:
            best_u, best_f = u, f
    theta = 10.0 ** best_u
    return theta, lg_eval(theta, st, T)


# --------------------------------------------------------------------------- #

def _fmt_theta(theta) -> str:
    return "  ".join(f"{n}={v:.4g}" for n, v in zip(PARAM_NAMES, theta))


def _load_full(path: str) -> dict:
    """Результаты полной симуляции (список записей {point, seed, total, blown, pnl})."""
    if not os.path.exists(path):
        return {}
    rows = json.load(open(path, encoding="utf-8"))
    out = {}
    for name in {r["point"] for r in rows}:
        rr = [r for r in rows if r["point"] == name]
        tot = [(-BLOWUP if r["blown"] else r["total"]) for r in rr]
        ok = [r for r in rr if not r["blown"]]
        out[name] = {"median": float(np.median(tot)), "mean": float(np.mean(tot)),
                     "std": float(np.std(tot)), "n_blown": len(rr) - len(ok), "n": len(rr),
                     "pnl": {a: (float(np.mean([r["pnl"][a] for r in ok])) if ok else 0.0)
                             for a in AGENTS}}
    return out


def main() -> None:
    T = T_RUN
    st = calibrate()
    print_calibration(st)
    rec = recorded_paths()
    syn = synth_paths(st, T, 20, seed=1)
    full = _load_full(sys.argv[1]) if len(sys.argv) > 1 else {}

    points = {
        "baseline":          (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        "opt_best":          (0.263, 0.0646, 0.02636, 4.301, 0.01115, 0.03708),
        "opt_best/100 h_r1": (0.263, 0.000646, 0.02636, 4.301, 0.01115, 0.03708),
        "T1 hedges too":     (0.263, 4.0, 0.02636, 4.301, 0.01115, 0.03708),
    }
    t0 = time.perf_counter()
    theta_star, lg_star = lg_optimize(st, T)
    print(f"\nАналитический оптимум (LG, {time.perf_counter()-t0:.1f}s): "
          f"E[PnL] = {T*lg_star['pi']:+.2f}, P(взрыв) ~ {lg_star['p_blow']:.2f}, "
          f"срыв 4ab = {lg_star['runaway']:.2f}, разряд/тик = {lg_star['discharge']:.2f}")
    print("  " + _fmt_theta(theta_star))
    points["LG optimum"] = tuple(float(v) for v in theta_star)
    with open(RESULT_FILE, "w", encoding="utf-8") as f:
        json.dump({"theta": dict(zip(PARAM_NAMES, points["LG optimum"])),
                   "E_pnl": T * lg_star["pi"], "p_blow": lg_star["p_blow"]}, f, indent=1)

    print()
    print(f"{'точка':<19}|{'LG: E[PnL]':>10} {'P(bl)':>5} {'rms n1/nA':>11}|"
          f"{'reduced, записанные пути':^32}|{'reduced, VAR(1) пути':^32}|{'полная симуляция':^32}")
    print(f"{'':<19}|{'':>28}|" + f"{'med':>8} {'mean':>8} {'std':>7} {'bl':>6}|" * 3)
    for name, theta in points.items():
        lg = lg_eval(theta, st, T)
        rr = reduced_eval(theta, st, rec)
        rs = reduced_eval(theta, st, syn)
        fs = full.get(name)
        def block(r):
            return (f"{r['median']:>+8.2f} {r['mean']:>+8.2f} {r['std']:>7.2f} "
                    f"{r['n_blown']:>3}/{r['n']:<2}") if r else f"{'—':^32}"
        print(f"{name:<19}|{T*lg['pi']:>+10.2f} {lg['p_blow']:>5.2f} "
              f"{lg['rms_n'][0]:>5.0f}/{lg['rms_n'][2]:<5.0f}|{block(rr)}|{block(rs)}|{block(fs)}")
        print(f"{'':<19}|  harvest {lg['harvest'][0]:+.4f}/{lg['harvest'][1]:+.4f}  cost "
              f"{lg['cost'][0]:.4f}/{lg['cost'][1]:.4f} за тик;  1/R={1/lg['R']:.0f}  "
              f"E|h|={lg['e_abs_h'][0]:.3f}/{lg['e_abs_h'][1]:.3f}  beta={lg['beta'][0]:.2f}/{lg['beta'][1]:.2f}"
              f"  risk={np.array2string(lg['risk'], precision=0, suppress_small=True)}"
              f"  срыв 4ab={lg['runaway']:.3f} разряд={lg['discharge']:.2f} поток rms={lg['flow_rms']:.2f}")
        print(f"{'':<19}|  reduced(зап.): оборот хеджа {rr['turn1']:.3f}/{rr['turn2']:.3f}, поток {rr['flow']:.3f} за тик, "
              f"rms n1={rr['rms_n1']:.0f};  PnL: " + "  ".join(f"{a}={rr['pnl'][a]:+.1f}" for a in AGENTS)
              + (("   полная: " + "  ".join(f"{a}={fs['pnl'][a]:+.1f}" for a in AGENTS)) if fs else ""))


if __name__ == "__main__":
    main()
