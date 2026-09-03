"""
opt_simulation.py — облегчённая симуляция для подбора гиперпараметров.

Отличия от agent_simulation.py:
    * популяция — ровно 4 агента, по одному на роль, у каждого свои параметры:
          T1 (h_m1, h_r1), T2 (h_m2, h_r2), AX (h_mx), AY (h_my);
    * нет Recorder, срезов стакана и отчётов — храним только equity агентов;
    * total_steps = 6000, эволюция фактически выключена (да и с одним агентом
      на тип она никого убить не может — последний живой в типе защищён);
    * точка входа — функция run_pnl(params) -> dict с PnL каждого агента и
      суммой; total_pnl(params) -> float для оптимизатора.

Основные функции одного тика (submit_translator_orders, submit_arb_orders,
hedge_translators, evolution_step) импортируются из agent_simulation без
изменений — логика симуляции идентична.

Запуск демо:  python opt_simulation.py
"""

from __future__ import annotations

import numpy as np

import agent_formulas as F
from coupled_market import CoupledMarket
from pfx_exchange import Exchange as PortfolioExchange
from agent_simulation import (
    Agent,
    EwmaVar,
    SimConfig,
    evolution_step,
    hedge_translators,
    submit_arb_orders,
    submit_translator_orders,
)

# порядок параметров в векторе оптимизации
PARAM_NAMES = ("h_m1", "h_r1", "h_m2", "h_r2", "h_mx", "h_my")


class _NullRecorder:
    """Заглушка вместо Recorder: hedge_translators зовёт rec.hedge, нам
    сама запись не нужна."""

    def hedge(self, *args, **kwargs) -> None:
        pass


def opt_config(**overrides) -> SimConfig:
    """SimConfig с настройками для оптимизации; overrides — поверх них."""
    base = dict(
        total_steps=6000,
        evolution_start=10 ** 9,   # эволюция не наступает
        book_snapshot_every=0,     # срезы стакана не нужны
        progress_every=0,
    )
    base.update(overrides)
    return SimConfig(**base)


def build_agents(params) -> list[Agent]:
    """4 агента с индивидуальными параметрами.

    params — последовательность (h_m1, h_r1, h_m2, h_r2, h_mx, h_my).
    """
    h_m1, h_r1, h_m2, h_r2, h_mx, h_my = (float(p) for p in params)
    return [
        Agent(name="T1", kind="T1", h_m=h_m1, h_r=h_r1),
        Agent(name="T2", kind="T2", h_m=h_m2, h_r=h_r2),
        Agent(name="AX", kind="AX", h_m=h_mx),
        Agent(name="AY", kind="AY", h_m=h_my),
    ]


def run_pnl(params, cfg: SimConfig | None = None, verbose: bool = False,
            return_equity: bool = False,
            blowup_limit: float | None = None) -> dict:
    """Прогон симуляции с заданными параметрами агентов.

    blowup_limit — защита от вырожденных режимов: если |PnL| любого агента
    превышает порог, прогон обрывается на этом тике (при экстремальных
    параметрах клиринг CE плохо обусловлен и цены идут вразнос — досчитывать
    такой прогон бессмысленно). None — не проверять.

    Возвращает dict:
        pnl        — {имя агента: PnL в X1 на момент окончания}
        total      — суммарный PnL четырёх агентов
        gamma      — откалиброванная gamma
        blown      — True, если прогон оборван по blowup_limit
        end_tick   — тик, на котором прогон закончился (T или тик обрыва)
        equity     — (4, T+1) траектории PnL (только при return_equity=True;
                     после end_tick остаются нули)
    """
    if cfg is None:
        cfg = opt_config()

    # --- рынки (в точности как в run_simulation) --------------------------- #
    market = CoupledMarket(
        cfg.venue1, cfg.venue2, tick_size=cfg.tick_size,
        initial_price=cfg.initial_price,
        anchor_ewma_half_life=cfg.anchor_half_life,
        depth_band=cfg.depth_band, seed=cfg.seed,
        fundamental_vol=cfg.fundamental_vol)
    market.warmup(cfg.warmup)
    ex1, ex2 = market.exchanges["1"], market.exchanges["2"]

    ce = PortfolioExchange(
        assets=list(F.ASSETS),
        prices={"X1": 1.0, "Y1": ex1.mid, "X2": 1.0, "Y2": ex2.mid},
        unit_of_account="X1")

    # --- калибровка gamma --------------------------------------------------- #
    if cfg.gamma is not None:
        gamma = cfg.gamma
    else:
        costs = [F.hedge_cost(e.spread, e.mid, cfg.tick_size)
                 for e in (ex1, ex2)]
        costs = [c for c in costs if c is not None]
        sigma2s = [e.volatility ** 2 for e in (ex1, ex2)]
        gamma = F.calibrate_gamma(
            cost=float(np.mean(costs)) if costs else 0.5 * cfg.tick_size / cfg.initial_price,
            sigma2=float(np.mean(sigma2s)),
            q_max_fraction=cfg.q_max_fraction)

    # --- агенты и история --------------------------------------------------- #
    agents = build_agents(params)
    n, T = len(agents), cfg.total_steps
    equity = np.zeros((n, T + 1))
    rec = _NullRecorder()

    basis_vol = {
        kind: EwmaVar(cfg.arb_vol_half_life, ce.rate(F.PORTFOLIOS[kind]))
        for kind in ("AX", "AY")
    }

    # --- основной цикл (как в run_simulation, без записи) ------------------- #
    blown = False
    end_tick = T
    for t in range(1, T + 1):
        market.step()
        submit_translator_orders(cfg, market, ce, agents, gamma)
        submit_arb_orders(cfg, ce, agents, gamma, basis_vol)
        ce.step(dt=1.0)
        hedge_translators(cfg, market, ce, agents, gamma, rec, t)

        for kind in ("AX", "AY"):
            basis_vol[kind].update(ce.rate(F.PORTFOLIOS[kind]))

        for i, a in enumerate(agents):
            equity[i, t] = (ce.mark_to_market(a.name) if a.active
                            else equity[i, t - 1])

        if (blowup_limit is not None
                and np.abs(equity[:, t]).max() > blowup_limit):
            blown = True
            end_tick = t
            break

        evolution_step(cfg, agents, equity, t)

        if verbose and cfg.progress_every and t % cfg.progress_every == 0:
            pnls = "  ".join(f"{a.name}={equity[i, t]:+9.4f}"
                             for i, a in enumerate(agents))
            print(f"t={t:>6}  {pnls}  total={equity[:, t].sum():+9.4f}")

    pnl = {a.name: float(equity[i, end_tick]) for i, a in enumerate(agents)}
    out = {"pnl": pnl, "total": float(sum(pnl.values())), "gamma": gamma,
           "blown": blown, "end_tick": end_tick}
    if return_equity:
        out["equity"] = equity
    return out


def total_pnl(params, cfg: SimConfig | None = None) -> float:
    """Суммарный PnL четырёх агентов — то, что максимизирует оптимизатор."""
    return run_pnl(params, cfg=cfg)["total"]


if __name__ == "__main__":
    demo = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    print("Демо-прогон:", dict(zip(PARAM_NAMES, demo)))
    res = run_pnl(demo, cfg=opt_config(progress_every=1000), verbose=True)
    print()
    for name, value in res["pnl"].items():
        print(f"  {name}: PnL = {value:+10.4f}")
    print(f"  сумма:     {res['total']:+10.4f}   (gamma={res['gamma']:.4g})")
