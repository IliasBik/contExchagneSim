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
       с наибольшим убытком за последние window шагов — но только если
       его суммарная прибыль за всё время тоже отрицательна и он не
       последний живой в своём типе. Деактивированный навсегда перестаёт
       торговать, его прибыль замораживается, доля мощности перетекает
       живым.

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
    evolution_step: int = 100
    window: int = 2000

    # --- запись прогона для отчёта ----------------------------------------- #
    # Пишется всегда: объём данных мал (десятки МБ на 12000 тиков), а строить
    # отчёт по неполной записи невозможно. Срезы стакана — единственное, что
    # заметно весит, поэтому берутся с прореживанием.
    book_snapshot_every: int = 25   # период срезов стакана, тиков (0 — не писать)
    book_band: float = 8.0          # полуширина полосы среза вокруг мида, цена
    book_bin: float = 0.25          # ширина ценового бина среза
    report_dir: str = "report"      # каталог отчёта (tex, pdf, фигуры, кэш)

    # --- рынок -------------------------------------------------------------#
    seed: int | None = 7
    warmup: int = 200
    tick_size: float = 0.01
    initial_price: float = 100.0
    # depth_band масштабируется вместе с price_std: полоса, в которой
    # считается глубина у мида, должна накрывать типичный разброс заявок
    depth_band: float = 4.0
    anchor_half_life: float = 20.0
    # волатильность фундаментальной цены: лог-шок якоря за тик; уровень
    # цен блуждает как sigma_F * sqrt(T) (1e-3 -> ~11% за 12000 тиков)
    fundamental_vol: float = 1e-3
    venue1: ExchangeConfig = field(default_factory=lambda: ExchangeConfig(
        name="1", arrival_rate=10.0, order_size=1.0, order_ttl=5,
        price_std=3.0, ewma_half_life=10.0))
    venue2: ExchangeConfig = field(default_factory=lambda: ExchangeConfig(
        name="2", arrival_rate=3.0, order_size=1.0, order_ttl=5,
        price_std=3.0, ewma_half_life=10.0))

    progress_every: int = 500    # период печати прогресса (0 — молча)


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
# Запись прогона
# --------------------------------------------------------------------------- #

KINDS = ("T1", "T2", "AX", "AY")
VENUES = ("1", "2")


class Recorder:
    """Полная потиковая запись прогона — сырьё для отчёта.

    Пишет три группы величин:

    МИР      — справедливая цена и якорь, по каждой лимитной бирже: мид,
               цена аукциона, лучшие котировки, спред, глубина у мида,
               EWMA-волатильность, объём аукциона; плюс прореженные срезы
               стакана (объём по бинам относительно мида) для тепловых карт.

    CE       — оборот за тик всего и в разрезе типов агентов, число активных
               заявок, ранг книги, невязка клиринга, среднее |f| по типам.

    АГЕНТЫ   — оборот каждого агента, и разложение его PnL на две компоненты.
               Разложение точное, а не оценочное: внутри тика цены CE меняются
               ровно один раз (в ce.step), поэтому

                   dPnL = sum_a q_prev[a] * (P_new[a] - P_old[a])   переоценка
                        + sum_a dq_сделки[a] * P_new[a]             = 0
                        + sum_a dq_хедж[a]   * P_new[a]             хедж

               Средний член равен нулю тождественно: портфель заявки
               самофинансируем (sum w = 0), поэтому сделка на CE не создаёт
               и не уничтожает стоимость в момент исполнения. Весь PnL
               транслятора — это переоценка накопленной позиции плюс то,
               что он потерял (или выиграл) на хедже об домашний стакан.
               Невязка этого тождества пишется в pnl_residual и служит
               проверкой корректности (должна быть машинным нулём).
    """

    def __init__(self, cfg: SimConfig, agents: list, ce: PortfolioExchange):
        n, T = len(agents), cfg.total_steps
        self.cfg = cfg
        self.n_assets = len(F.ASSETS)
        self._idx = {a.name: i for i, a in enumerate(agents)}
        self._kind_idx = {k: j for j, k in enumerate(KINDS)}
        self._agent_kind = np.array([self._kind_idx[a.kind] for a in agents])

        # --- агенты ---------------------------------------------------------#
        self.turnover = np.zeros((n, T + 1), np.float32)   # |нотионал| на CE
        self.reval = np.zeros((n, T + 1), np.float32)      # переоценка позиции
        self.hedge_pnl = np.zeros((n, T + 1), np.float32)  # результат хеджа
        self.pnl_residual = np.zeros(T + 1)                # контроль тождества

        # --- мир ------------------------------------------------------------#
        self.fundamental = np.zeros(T + 1)
        self.anchor = np.zeros(T + 1)
        self.mid = np.zeros((T + 1, 2))
        self.last_price = np.zeros((T + 1, 2))
        self.best_bid = np.full((T + 1, 2), np.nan)
        self.best_ask = np.full((T + 1, 2), np.nan)
        self.spread = np.full((T + 1, 2), np.nan)
        self.depth = np.zeros((T + 1, 2))
        self.vol = np.zeros((T + 1, 2))
        self.volume = np.zeros((T + 1, 2))

        # --- CE --------------------------------------------------------------#
        self.ce_gross = np.zeros(T + 1)
        self.ce_gross_kind = np.zeros((T + 1, len(KINDS)))
        self.ce_absf_kind = np.zeros((T + 1, len(KINDS)))
        self.ce_orders = np.zeros(T + 1, np.int32)
        self.ce_rank = np.zeros(T + 1, np.int8)
        self.ce_imbalance = np.zeros(T + 1)

        # --- хедж (по площадкам) ----------------------------------------------#
        self.hedge_count = np.zeros((T + 1, 2))
        self.hedge_notional = np.zeros((T + 1, 2))
        self.hedge_result = np.zeros((T + 1, 2))   # выигрыш(+)/потеря(-), X1

        # --- срезы стакана ----------------------------------------------------#
        step = cfg.book_bin
        self.book_edges = np.arange(-cfg.book_band, cfg.book_band + step, step)
        self.book_ticks: list[int] = []
        self._book_bid: list[np.ndarray] = []
        self._book_ask: list[np.ndarray] = []

        # состояние для разложения PnL: количества и цены до клиринга
        self._q_prev = np.zeros((n, self.n_assets))
        self._p_prev = np.array([ce.price(a) for a in F.ASSETS])
        self.stats = {"hedge_count": 0, "hedge_value": 0.0}

    # ------------------------------------------------------------------ мир

    def market(self, t: int, market: CoupledMarket) -> None:
        """Состояние обеих лимитных бирж после их тика."""
        self.fundamental[t] = market.fundamental
        self.anchor[t] = market.anchor
        for v, name in enumerate(VENUES):
            ex = market.exchanges[name]
            self.mid[t, v] = ex.mid
            self.last_price[t, v] = ex.last_price
            self.volume[t, v] = ex.last_trade_volume
            self.vol[t, v] = ex.volatility
            self.depth[t, v] = ex.depth_near_mid(self.cfg.depth_band)
            if ex.best_bid is not None:
                self.best_bid[t, v] = ex.best_bid
            if ex.best_ask is not None:
                self.best_ask[t, v] = ex.best_ask
            if ex.spread is not None:
                self.spread[t, v] = ex.spread

        every = self.cfg.book_snapshot_every
        if every and t % every == 0:
            self.book_ticks.append(t)
            for v, name in enumerate(VENUES):
                ex = market.exchanges[name]
                self._book_bid.append(self._bin_orders(ex.bids, ex.mid))
                self._book_ask.append(self._bin_orders(ex.asks, ex.mid))

    def _bin_orders(self, orders, mid: float) -> np.ndarray:
        """Объём заявок по бинам смещения цены от мида."""
        out = np.zeros(len(self.book_edges) - 1)
        if not orders:
            return out
        offsets = np.fromiter((o.price - mid for o in orders), float, len(orders))
        sizes = np.fromiter((o.size for o in orders), float, len(orders))
        idx = np.digitize(offsets, self.book_edges) - 1
        ok = (idx >= 0) & (idx < out.size)
        np.add.at(out, idx[ok], sizes[ok])
        return out

    # ------------------------------------------------------------------- CE

    def before_clearing(self, ce: PortfolioExchange, agents: list) -> None:
        """Количества и цены до клиринга — база для разложения PnL."""
        self._p_prev = np.array([ce.price(a) for a in F.ASSETS])
        for i, a in enumerate(agents):
            bal = ce.balances.get(a.name, {})
            for j, asset in enumerate(F.ASSETS):
                self._q_prev[i, j] = bal.get(asset, 0.0)

    def clearing(self, t: int, report) -> None:
        """Итоги клиринга: оборот всего, по агентам и по типам."""
        self.ce_gross[t] = report.gross_notional
        self.ce_orders[t] = len(report.fills)
        self.ce_rank[t] = report.rank
        self.ce_imbalance[t] = report.max_value_imbalance
        counts = np.zeros(len(KINDS))
        for fill in report.fills:
            i = self._idx.get(fill.agent)
            if i is None:
                continue
            k = self._agent_kind[i]
            self.turnover[i, t] += abs(fill.notional)
            self.ce_gross_kind[t, k] += abs(fill.notional)
            self.ce_absf_kind[t, k] += abs(fill.f)
            counts[k] += 1.0
        np.divide(self.ce_absf_kind[t], counts, out=self.ce_absf_kind[t],
                  where=counts > 0)

    def hedge(self, t: int, venue: str, agent_idx: int, notional: float,
              result_value: float) -> None:
        """Один хедж: нотионал по миду и его результат в единицах счёта."""
        v = VENUES.index(venue)
        self.hedge_count[t, v] += 1.0
        self.hedge_notional[t, v] += notional
        self.hedge_result[t, v] += result_value
        self.hedge_pnl[agent_idx, t] += result_value
        self.stats["hedge_count"] += 1
        self.stats["hedge_value"] += notional

    def after_tick(self, t: int, ce: PortfolioExchange, agents: list,
                   equity: np.ndarray) -> None:
        """Переоценка позиций и проверка тождества разложения PnL."""
        p_new = np.array([ce.price(a) for a in F.ASSETS])
        alive = np.array([a.active for a in agents], dtype=bool)
        self.reval[:, t] = np.where(
            alive, self._q_prev @ (p_new - self._p_prev), 0.0)
        delta = equity[:, t] - equity[:, t - 1]
        parts = self.reval[:, t] + self.hedge_pnl[:, t]
        self.pnl_residual[t] = float(np.max(np.abs(delta - parts)))

    # ----------------------------------------------------------------- сборка

    def finish(self) -> None:
        """Срезы стакана — в массивы (снимки, площадка, бин)."""
        n_snap = len(self.book_ticks)
        shape = (n_snap, 2, len(self.book_edges) - 1)
        self.book_bid = (np.array(self._book_bid).reshape(shape) if n_snap
                         else np.zeros(shape))
        self.book_ask = (np.array(self._book_ask).reshape(shape) if n_snap
                         else np.zeros(shape))
        self.book_ticks = np.array(self.book_ticks, dtype=int)
        self._book_bid, self._book_ask = [], []


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
        venue, y_asset, cash_asset = F.TRANSLATOR_HOME[kind]
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
            # lam посчитана в кэше домашней площадки (H = depth*mid, а mid
            # у T2 выражен в X2); lam_ccy поручает бирже конвертацию в
            # единицу счёта X1 по цене прошлого шага
            ce.submit(Order(weights=F.PORTFOLIOS[kind], z=z, lam=lam,
                            agent=a.name, lam_ccy=cash_asset,
                            expiry=ce.t + 1.0))


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
                      gamma: float, rec: Recorder, t: int) -> None:
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
        for i, a in enumerate(agents):
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
            dq_y = sign * filled
            dq_cash = -sign * result["total_cost"]
            # стоимость, внесённая хеджем в портфель агента (в единицах счёта):
            # актив ушёл по встречной стороне книги, а не по оценке CE, поэтому
            # величина обычно отрицательна и равна потере на полуспреде
            value = dq_y * ce.price(y_asset) + dq_cash * ce.price(cash_asset)
            ce.deposit(a.name, y_asset, dq_y)
            ce.deposit(a.name, cash_asset, dq_cash)
            rec.hedge(t, venue, i, filled * mid, value)


def evolution_step(cfg: SimConfig, agents: list[Agent],
                   equity: np.ndarray, t: int) -> Agent | None:
    """Деактивация худшего агента по убытку за последние window шагов.

    Кандидаты перебираются от худшего окна к лучшему; выбывает первый,
    у кого одновременно (а) убыток за окно строго отрицателен,
    (б) суммарная прибыль за всё время тоже отрицательна — просадка
    долгосрочно прибыльного агента не повод его убивать, и
    (в) он не последний живой в своём типе.
    Не более одного выбытия за evolution_step тиков.
    """
    if t <= cfg.evolution_start:
        return None
    if t % cfg.evolution_step != 0:
        return None

    window = cfg.window
    alive = [i for i, a in enumerate(agents) if a.active]
    kind_counts = Counter(agents[i].kind for i in alive)
    alive.sort(key=lambda i: equity[i, t] - equity[i, t - window])
    for i in alive:
        pnl = equity[i, t] - equity[i, t - window]
        if pnl >= 0.0:
            break                          # убыточных за окно больше нет
        a = agents[i]
        if equity[i, t] >= 0.0:
            continue                       # суммарно прибыльный — защищён
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
    rec: Recorder             # полная потиковая запись (см. Recorder)


def run_simulation(cfg: SimConfig, verbose: bool = True) -> SimResult:
    # --- рынки ------------------------------------------------------------ #
    market = CoupledMarket(
        cfg.venue1, cfg.venue2, tick_size=cfg.tick_size,
        initial_price=cfg.initial_price,
        anchor_ewma_half_life=cfg.anchor_half_life,
        depth_band=cfg.depth_band, seed=cfg.seed,
        fundamental_vol=cfg.fundamental_vol)
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
    rec = Recorder(cfg, agents, ce)
    rec.market(0, market)
    stats = rec.stats

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
        rec.market(t, market)
        submit_translator_orders(cfg, market, ce, agents, gamma)
        submit_arb_orders(cfg, ce, agents, gamma, basis_vol)
        rec.before_clearing(ce, agents)
        report = ce.step(dt=1.0)
        rec.clearing(t, report)
        hedge_translators(cfg, market, ce, agents, gamma, rec, t)

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
        rec.after_tick(t, ce, agents, equity)

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

    rec.finish()
    return SimResult(config=cfg, agents=agents, equity=equity,
                     inventory=inventory, deaths=deaths, ce_prices=ce_prices,
                     mids=mids, gross=gross, gamma=gamma, stats=stats,
                     market=market, ce=ce, rec=rec)


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
