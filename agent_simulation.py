"""
agent_simulation.py — экосистема агентов поверх двух рынков.

Связывает три части:
    coupled_market.py — две лимитные биржи с коррелированными ценами
                        (биржа "1" — толстая, "2" — тонкая);
    pfx_exchange.py   — непрерывная биржа портфельных заявок (CE) с четырьмя
                        активами X1, Y1, X2, Y2; единица счёта — X1;
    agent_formulas.py — все формулы поведения агентов (см. модуль F).

Порядок одного тика симуляции:
    1) market.step()      — фоновый поток и аукционы на лимитных биржах;
    2) агенты читают свои наблюдаемые величины и перевыставляют заявки на CE
       (заявка живёт ровно один клиринг: expiry = t + 1);
    3) ce.step()          — клиринг CE, филлы ложатся в балансы;
    4) трансляторы хеджируют излишек инвентаря об домашний стакан
       ("бумажный" хедж: quote_hedge оценивает исполнение, стакан не двигая,
        результат заносится в балансы CE через deposit);
    5) фиксация прибыли (mark-to-market в X1) каждого агента;
    6) эволюция: после evolution_start каждый тик деактивируется агент
       с наибольшим убытком за последние window шагов, если этот убыток
       отрицателен и агент не последний живой в своём типе. Деактивированный
       навсегда перестаёт торговать, его прибыль замораживается, доля
       мощности перетекает живым.

Запуск демо:  python agent_simulation.py
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

import agent_formulas as F
from coupled_market import CoupledMarket, ExchangeConfig
from pfx_exchange import Exchange as PortfolioExchange, Order


# --------------------------------------------------------------------------- #
# Конфигурация эксперимента
# --------------------------------------------------------------------------- #

@dataclass
class SimConfig:
    # --- население: сетки гиперпараметров -------------------------------- #
    # трансляторы: все комбинации h_m x h_R (4x4 = 16 на площадку);
    # арбитражёры: по arb_copies копий на каждое h_m (4x4 = 16 на тип)
    h_m_values: tuple = (0.55, 0.8, 1.0, 1.35)
    h_r_values: tuple = (0.55, 0.8, 1.0, 1.35)
    arb_copies: int = 4

    # --- капитал и торговая мощность ------------------------------------- #
    # kappa подобраны так, чтобы обороты CE были заметны на фоне книг, а
    # семейство трансляторов тонкой биржи не задавливалось арбитражёрами
    c0: float = 1000.0        # стартовый виртуальный капитал, X1
    kappa_t: float = 1.0      # мощность семейства трансляторов, 1/тик
    kappa_a: float = 0.01     # мощность арбитражёра от капитала, 1/тик

    # --- риск ------------------------------------------------------------- #
    gamma: float | None = None    # неприятие риска; None -> калибровка
    q_max_fraction: float = 0.005  # хедж-порог как доля капитала (калибровка)
    capital_floor_frac: float = 0.01  # пол капитала в g, доля C0
    shading_clamp: float = 1.0    # предохранитель |g*q| в экспоненте
    arb_shading: bool = True      # шейдинг котировки арбитражёров
    arb_vol_half_life: float = 20.0  # полураспад EWMA-волатильности базиса

    # --- эволюционный отбор ------------------------------------------------#
    total_steps: int = 12000
    evolution_start: int = 6000
    window: int = 2000

    # --- рынок -------------------------------------------------------------#
    seed: int | None = 7
    warmup: int = 200
    tick_size: float = 0.01
    initial_price: float = 100.0
    depth_band: float = 0.5
    anchor_half_life: float = 20.0
    venue1: ExchangeConfig = field(default_factory=lambda: ExchangeConfig(
        name="1", arrival_rate=40.0, order_size=1.0, order_ttl=20,
        price_std=0.5, ewma_half_life=20.0))
    venue2: ExchangeConfig = field(default_factory=lambda: ExchangeConfig(
        name="2", arrival_rate=8.0, order_size=1.0, order_ttl=20,
        price_std=0.5, ewma_half_life=20.0))

    progress_every: int = 2000    # период печати прогресса (0 — молча)


# --------------------------------------------------------------------------- #
# Вспомогательное
# --------------------------------------------------------------------------- #

class EwmaVar:
    """EWMA-дисперсия лог-приростов уровня (та же схема, что у лимитных бирж).

    Арбитражёр не видит лимитных бирж, поэтому волатильность своего базиса
    он оценивает сам — по ценам CE.
    """

    def __init__(self, half_life: float, level0: float):
        self._alpha = 1.0 - 0.5 ** (1.0 / half_life)
        self.var = 0.0
        self._prev = level0

    def update(self, level: float) -> None:
        r = np.log(level / self._prev)
        self.var = (1.0 - self._alpha) * self.var + self._alpha * r * r
        self._prev = level


# --------------------------------------------------------------------------- #
# Агенты
# --------------------------------------------------------------------------- #

@dataclass
class Agent:
    """Участник экосистемы: параметры и эволюционное состояние.

    Портфель заявки, домашняя площадка и торгуемые ноги определяются типом
    (kind) через таблицы в agent_formulas. Инвентарь агента живёт в балансах
    CE (единственный источник правды), прибыль — mark-to-market в X1.
    """

    name: str
    kind: str                 # "T1" | "T2" | "AX" | "AY"
    h_m: float
    h_r: float | None = None  # только у трансляторов
    active: bool = True
    death_tick: int | None = None


def build_population(cfg: SimConfig) -> list[Agent]:
    """Популяция: 16 T1 + 16 T2 + 16 AX + 16 AY при сетках по умолчанию."""
    agents: list[Agent] = []
    for kind in ("T1", "T2"):
        for hm in cfg.h_m_values:
            for hr in cfg.h_r_values:
                agents.append(Agent(name=f"{kind}[m={hm:g},r={hr:g}]",
                                    kind=kind, h_m=hm, h_r=hr))
    for kind in ("AX", "AY"):
        for hm in cfg.h_m_values:
            for copy in range(cfg.arb_copies):
                agents.append(Agent(name=f"{kind}[m={hm:g}]#{copy + 1}",
                                    kind=kind, h_m=hm))
    return agents


# --------------------------------------------------------------------------- #
# Один тик: котирование, хедж, эволюция
# --------------------------------------------------------------------------- #

def submit_translator_orders(cfg: SimConfig, market: CoupledMarket,
                             ce: PortfolioExchange, agents: list[Agent],
                             gamma: float) -> None:
    """Трансляторы читают домашний стакан и перевыставляют заявки на CE."""
    floor = cfg.capital_floor_frac * cfg.c0
    for kind in ("T1", "T2"):
        venue, y_asset, _cash = F.TRANSLATOR_HOME[kind]
        exv = market.exchanges[venue]
        members = [a for a in agents if a.kind == kind and a.active]
        if not members:
            continue

        # наблюдаемые семейством величины
        mid, spread = exv.mid, exv.spread
        sigma2 = exv.volatility ** 2
        depth = exv.depth_near_mid(cfg.depth_band)
        H = F.book_quality(depth, mid, spread, cfg.tick_size)
        if H <= 0.0:
            continue                      # пустая/односторонняя книга

        capitals = [F.capital(ce.mark_to_market(a.name), cfg.c0)
                    for a in members]
        for a, cap in zip(members, capitals):
            lam = F.lam_translator(cfg.kappa_t, H, F.share(cap, capitals),
                                   a.h_m)
            if lam <= 0.0:
                continue
            g = F.risk_coefficient(a.h_r, gamma, sigma2, cap, floor)
            q_units = ce.balances.get(a.name, {}).get(y_asset, 0.0)
            z = F.shaded_quote(mid, g, q_units * mid, cfg.shading_clamp)
            ce.submit(Order(weights=F.PORTFOLIOS[kind], z=z, lam=lam,
                            agent=a.name, expiry=ce.t + 1.0))


def submit_arb_orders(cfg: SimConfig, ce: PortfolioExchange,
                      agents: list[Agent], gamma: float,
                      basis_vol: dict[str, EwmaVar]) -> None:
    """Арбитражёры выставляют кросс-заявки на CE (безразличие около 1)."""
    floor = cfg.capital_floor_frac * cfg.c0
    for kind in ("AX", "AY"):
        members = [a for a in agents if a.kind == kind and a.active]
        if not members:
            continue
        sigma2 = basis_vol[kind].var
        long_leg = F.ARB_LONG_LEG[kind]
        for a in members:
            cap = F.capital(ce.mark_to_market(a.name), cfg.c0)
            lam = F.lam_arb(cfg.kappa_a, cap, a.h_m)
            if lam <= 0.0:
                continue
            z = 1.0
            if cfg.arb_shading:
                g = F.risk_coefficient(1.0, gamma, sigma2, cap, floor)
                q_value = (ce.balances.get(a.name, {}).get(long_leg, 0.0)
                           * ce.price(long_leg))
                z = F.shaded_quote(1.0, g, q_value, cfg.shading_clamp)
            ce.submit(Order(weights=F.PORTFOLIOS[kind], z=z, lam=lam,
                            agent=a.name, expiry=ce.t + 1.0))


def hedge_translators(cfg: SimConfig, market: CoupledMarket,
                      ce: PortfolioExchange, agents: list[Agent],
                      gamma: float, stats: dict) -> None:
    """Хедж излишка инвентаря об домашний стакан (одно обращение на агента).

    Решение принимается по наблюдаемым величинам (риск против полуспреда),
    исполнение — единственный вызов quote_hedge; его результат заносится
    в балансы CE двумя ногами. Потеря на хедже не проводится отдельно —
    она сама проявляется в PnL, потому что актив скинут хуже мида.
    """
    floor = cfg.capital_floor_frac * cfg.c0
    for kind in ("T1", "T2"):
        venue, y_asset, cash_asset = F.TRANSLATOR_HOME[kind]
        exv = market.exchanges[venue]
        mid = exv.mid
        cost = F.hedge_cost(exv.spread, mid, cfg.tick_size)
        if cost is None:
            continue                      # нет встречной стороны — не хеджируем
        sigma2 = exv.volatility ** 2
        for a in agents:
            if a.kind != kind or not a.active:
                continue
            q_units = ce.balances.get(a.name, {}).get(y_asset, 0.0)
            if abs(q_units) * mid < 1e-9:
                continue
            cap = F.capital(ce.mark_to_market(a.name), cfg.c0)
            g = F.risk_coefficient(a.h_r, gamma, sigma2, cap, floor)
            excess_value = F.hedge_excess_value(g, q_units * mid, cost)
            if excess_value <= 0.0:
                continue
            quantity = excess_value / mid
            side = "sell" if q_units > 0 else "buy"
            result = market.quote_hedge(venue, side, quantity)
            filled = result["filled"]
            if filled <= 0.0:
                continue
            sign = -1.0 if side == "sell" else 1.0
            ce.deposit(a.name, y_asset, sign * filled)
            ce.deposit(a.name, cash_asset, -sign * result["total_cost"])
            stats["hedge_count"] += 1
            stats["hedge_value"] += filled * mid


def evolution_step(cfg: SimConfig, agents: list[Agent],
                   equity: np.ndarray, t: int) -> Agent | None:
    """Деактивация худшего агента по убытку за последние window шагов.

    Кандидаты перебираются от худшего к лучшему; выбывает первый, у кого
    убыток строго отрицателен и кто не последний живой в своём типе.
    Не более одного выбытия за тик.
    """
    if t <= cfg.evolution_start:
        return None
    window = cfg.window
    alive = [i for i, a in enumerate(agents) if a.active]
    kind_counts = Counter(agents[i].kind for i in alive)
    alive.sort(key=lambda i: equity[i, t] - equity[i, t - window])
    for i in alive:
        pnl = equity[i, t] - equity[i, t - window]
        if pnl >= 0.0:
            break                          # убыточных больше нет
        a = agents[i]
        if kind_counts[a.kind] <= 1:
            continue                       # последний в типе — защищён
        a.active = False
        a.death_tick = t
        return a
    return None


# --------------------------------------------------------------------------- #
# Прогон симуляции
# --------------------------------------------------------------------------- #

@dataclass
class SimResult:
    config: SimConfig
    agents: list[Agent]
    equity: np.ndarray        # (n_agents, T+1): PnL каждого агента в X1
    inventory: np.ndarray     # (n_agents, T+1): стоимость позиции по торгуемой
                              # ноге, X1 (у T — Y-нога, у арбов — длинная нога)
    deaths: list[tuple]       # (тик, имя, убыток за окно)
    ce_prices: np.ndarray     # (T+1, 4): цены CE в порядке F.ASSETS
    mids: np.ndarray          # (T+1, 2): миды лимитных бирж
    gross: np.ndarray         # (T+1,): оборот CE за тик
    gamma: float
    stats: dict
    market: CoupledMarket
    ce: PortfolioExchange


def run_simulation(cfg: SimConfig, verbose: bool = True) -> SimResult:
    # --- рынки ------------------------------------------------------------ #
    market = CoupledMarket(
        cfg.venue1, cfg.venue2, tick_size=cfg.tick_size,
        initial_price=cfg.initial_price,
        anchor_ewma_half_life=cfg.anchor_half_life,
        depth_band=cfg.depth_band, seed=cfg.seed)
    market.warmup(cfg.warmup)
    ex1, ex2 = market.exchanges["1"], market.exchanges["2"]

    # CE стартует от состояния лимитных бирж после прогрева
    ce = PortfolioExchange(
        assets=list(F.ASSETS),
        prices={"X1": 1.0, "Y1": ex1.mid, "X2": 1.0, "Y2": ex2.mid},
        unit_of_account="X1")

    # --- калибровка gamma по состоянию после прогрева ---------------------- #
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

    # --- агенты и история --------------------------------------------------#
    agents = build_population(cfg)
    n, T = len(agents), cfg.total_steps
    equity = np.zeros((n, T + 1))
    inventory = np.zeros((n, T + 1))
    ce_prices = np.zeros((T + 1, len(F.ASSETS)))
    mids = np.zeros((T + 1, 2))
    gross = np.zeros(T + 1)
    ce_prices[0] = [ce.price(asset) for asset in F.ASSETS]
    mids[0] = (ex1.mid, ex2.mid)
    deaths: list[tuple] = []
    stats = {"hedge_count": 0, "hedge_value": 0.0}

    # волатильность базисов для шейдинга арбитражёров (оценка по ценам CE)
    basis_vol = {
        kind: EwmaVar(cfg.arb_vol_half_life, ce.rate(F.PORTFOLIOS[kind]))
        for kind in ("AX", "AY")
    }

    if verbose:
        _print_startup(cfg, gamma, ex1, ex2)

    # --- основной цикл ------------------------------------------------------#
    for t in range(1, T + 1):
        market.step()
        submit_translator_orders(cfg, market, ce, agents, gamma)
        submit_arb_orders(cfg, ce, agents, gamma, basis_vol)
        report = ce.step(dt=1.0)
        hedge_translators(cfg, market, ce, agents, gamma, stats)

        for kind in ("AX", "AY"):
            basis_vol[kind].update(ce.rate(F.PORTFOLIOS[kind]))

        for i, a in enumerate(agents):
            if not a.active:
                equity[i, t] = equity[i, t - 1]
                inventory[i, t] = inventory[i, t - 1]
                continue
            equity[i, t] = ce.mark_to_market(a.name)
            leg = (F.TRANSLATOR_HOME[a.kind][1] if a.kind in F.TRANSLATOR_HOME
                   else F.ARB_LONG_LEG[a.kind])
            inventory[i, t] = (ce.balances.get(a.name, {}).get(leg, 0.0)
                               * ce.price(leg))
        ce_prices[t] = [ce.price(asset) for asset in F.ASSETS]
        mids[t] = (ex1.mid, ex2.mid)
        gross[t] = report.gross_notional

        dead = evolution_step(cfg, agents, equity, t)
        if dead is not None:
            loss = equity[agents.index(dead), t] \
                 - equity[agents.index(dead), t - cfg.window]
            deaths.append((t, dead.name, loss))

        if verbose and cfg.progress_every and t % cfg.progress_every == 0:
            alive = Counter(a.kind for a in agents if a.active)
            print(f"t={t:>6}  Y1={ce.price('Y1'):8.3f}  Y2={ce.price('Y2'):8.3f}  "
                  f"X2={ce.price('X2'):7.4f}  mid1={ex1.mid:8.3f}  "
                  f"mid2={ex2.mid:8.3f}  живы: "
                  + " ".join(f"{k}:{alive.get(k, 0):>2}"
                             for k in ("T1", "T2", "AX", "AY")))

    return SimResult(config=cfg, agents=agents, equity=equity,
                     inventory=inventory, deaths=deaths, ce_prices=ce_prices,
                     mids=mids, gross=gross, gamma=gamma, stats=stats,
                     market=market, ce=ce)


# --------------------------------------------------------------------------- #
# Отчёты
# --------------------------------------------------------------------------- #

def _print_startup(cfg: SimConfig, gamma: float, ex1, ex2) -> None:
    print("=" * 78)
    print("Стартовая диагностика (после прогрева)")
    for name, exv in (("1", ex1), ("2", ex2)):
        depth = exv.depth_near_mid(cfg.depth_band)
        H = F.book_quality(depth, exv.mid, exv.spread, cfg.tick_size)
        print(f"  биржа {name}: mid={exv.mid:8.3f}  spread={exv.spread}  "
              f"depth={depth:7.1f}  H={H:10.1f}  "
              f"Lambda_T={cfg.kappa_t * H:8.2f}  vol={exv.volatility:.2e}")
    print(f"  gamma={gamma:.4g}   lambda арбитражёра при C0: "
          f"{cfg.kappa_a * cfg.c0:.2f}   хедж-порог: "
          f"{cfg.q_max_fraction * cfg.c0:.0f} X1 "
          f"(~{cfg.q_max_fraction * cfg.c0 / ex1.mid:.2f} шт Y)")
    print("=" * 78)


def print_final_report(res: SimResult) -> None:
    cfg, agents, equity = res.config, res.agents, res.equity
    T = cfg.total_steps
    print()
    print("=" * 78)
    survivors = sum(a.active for a in agents)
    print(f"Итог: {T} шагов, gamma={res.gamma:.4g}, выжило {survivors} из "
          f"{len(agents)}, выбыло {len(res.deaths)}; хеджей "
          f"{res.stats['hedge_count']} на {res.stats['hedge_value']:.1f} X1; "
          f"средний оборот CE {res.gross[1:].mean():.3f} X1/тик")
    print("=" * 78)

    for kind in ("T1", "T2", "AX", "AY"):
        rows = [(i, a) for i, a in enumerate(agents) if a.kind == kind]
        rows.sort(key=lambda r: equity[r[0], T], reverse=True)
        print(f"\n--- {kind} " + "-" * 66)
        for i, a in rows:
            status = "жив " if a.active else f"умер@{a.death_tick}"
            hr = f" h_R={a.h_r:<5g}" if a.h_r is not None else ""
            print(f"  {a.name:<22} h_m={a.h_m:<5g}{hr} {status:>10}  "
                  f"PnL={equity[i, T]:+10.4f}")
        # усреднение по значениям h_m — сигнал "инференса" гиперпараметров
        by_hm = {}
        for i, a in rows:
            by_hm.setdefault(a.h_m, []).append(equity[i, T])
        summary = "  ".join(f"h_m={hm:g}: {np.mean(v):+.4f}"
                            for hm, v in sorted(by_hm.items()))
        print(f"  средний PnL по h_m:  {summary}")
        if kind in ("T1", "T2"):
            by_hr = {}
            for i, a in rows:
                by_hr.setdefault(a.h_r, []).append(equity[i, T])
            summary = "  ".join(f"h_R={hr:g}: {np.mean(v):+.4f}"
                                for hr, v in sorted(by_hr.items()))
            print(f"  средний PnL по h_R:  {summary}")

    if res.deaths:
        print(f"\nПервые выбывшие: " + ", ".join(
            f"{name} (t={tick}, {loss:+.3f})"
            for tick, name, loss in res.deaths[:5]))


if __name__ == "__main__":
    result = run_simulation(SimConfig())
    print_final_report(result)
