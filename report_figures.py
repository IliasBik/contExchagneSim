"""
report_figures.py — все фигуры отчёта.

Две группы:

    ДИДАКТИЧЕСКИЕ (ex_*)   — маленькие самостоятельные примеры, объясняющие
                             механику: батч-аукцион, рождение стакана, связь
                             площадок, клиринг CE, шейдинг котировки, хедж.
                             Считаются на месте, на игрушечных данных или на
                             коротких вспомогательных прогонах.

    АНАЛИТИЧЕСКИЕ (fig_*)  — 13 фигур по основному прогону: мир, непрерывная
                             биржа, агенты. Каждая — 3–4 панели.

Никаких настроек здесь нет: всё, что влияет на картину, берётся из
конфигурации прогона (SimConfig) и из самих записанных рядов.

Фигуры сохраняются в PDF (вектор), размер подобран под ширину полосы
документа, поэтому в LaTeX они вставляются как есть, без масштабирования.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import agent_formulas as F
import report_data as D
from coupled_market import CoupledMarket, ExchangeConfig
from pfx_exchange import Exchange as PortfolioExchange, Order

# --------------------------------------------------------------------------- #
# Оформление
# --------------------------------------------------------------------------- #

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "legend.fontsize": 7,
    "legend.framealpha": 0.85,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "lines.linewidth": 1.0,
    "figure.dpi": 130,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

WIDTH = 6.9                      # ширина фигуры, дюймы (~17.5 см)
KIND_COLOR = {"T1": "#1f77b4", "T2": "#ff7f0e",
              "AX": "#2ca02c", "AY": "#d62728"}
VENUE_COLOR = ("#1f77b4", "#ff7f0e")
FUND_COLOR = "#111111"
ANCHOR_COLOR = "#888888"
CE_COLOR = "#2ca02c"
PHASE_COLOR = {"прогрев": "#dfe8f5", "отбор": "#f7e6d5",
               "стационар": "#dff0e2"}


@dataclass
class FigureSpec:
    """Одна фигура отчёта: файл, заголовок и подпись под ней."""

    key: str
    title: str
    caption: str


def _fig(nrows: int, ncols: int, height: float):
    fig, axes = plt.subplots(nrows, ncols, figsize=(WIDTH, height),
                             constrained_layout=True)
    return fig, np.atleast_1d(axes).ravel()


def _phase_spans(ax, ph: D.Phases, legend: bool = False) -> None:
    """Заливка фаз прогона под графиком по времени."""
    for label, (a, b) in ph.named:
        if b > a:
            ax.axvspan(a, b, color=PHASE_COLOR[label], zorder=0)
    if legend:
        handles = [Patch(facecolor=PHASE_COLOR[label], label=label)
                   for label, (a, b) in ph.named if b > a]
        ax.legend(handles=handles, loc="upper left", ncol=3)


def _thousands(ax) -> None:
    """Ось времени: неплотная сетка подписей с разделителем тысяч."""
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=5,
                                                             integer=True))
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}".replace(",", " ")))


def _smooth_window(run: D.RunData) -> int:
    """Окно сглаживания: сотая часть прогона, но не меньше 20 тиков."""
    return max(20, run.T // 100)


SHORT_PHASE = {"прогрев": "прогрев", "отбор": "отбор", "стационар": "стац."}


def _block_median(x: np.ndarray, win: int):
    """Медиана по последовательным блокам: устойчивая сводка ряда с выбросами.

    Для порога хеджа среднее не годится: порог обратно пропорционален
    дисперсии, поэтому редкие спокойные тики утягивают среднее вверх и
    прячут именно те провалы, в которые хедж и срабатывает.
    """
    win = max(int(win), 1)
    n = (len(x) // win) * win
    blocks = np.asarray(x[:n], dtype=float).reshape(-1, win)
    centers = np.arange(len(blocks)) * win + win / 2
    return centers, np.nanmedian(blocks, axis=1)


def _box(ax, data: list, labels: list, colors: list) -> None:
    bp = ax.boxplot(data, tick_labels=labels, showfliers=False,
                    patch_artist=True, widths=0.6,
                    medianprops={"color": "black", "linewidth": 1.0})
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)


# --------------------------------------------------------------------------- #
# ДИДАКТИЧЕСКИЕ ПРИМЕРЫ
# --------------------------------------------------------------------------- #

def ex_auction(run: D.RunData, ph: D.Phases):
    """Батч-аукцион: как из пересечения заявок получается единая цена."""
    from coupled_market import Exchange as LimitExchange

    cfg = run.cfg
    # книга после прогрева плюс заявки одного тика — ровно то, что видит
    # аукцион в реальном тике симуляции
    market = CoupledMarket(cfg.venue1, cfg.venue2, tick_size=cfg.tick_size,
                           initial_price=cfg.initial_price,
                           anchor_ewma_half_life=cfg.anchor_half_life,
                           depth_band=cfg.depth_band, seed=3)
    market.warmup(200)
    ex = market.exchanges["1"]
    ex.expire_orders(market.tick + 1)
    orders = ex.generate_background_orders(market.anchor, market.tick + 1)
    buys = ex.bids + [o for o in orders if o.side == "buy"]
    sells = ex.asks + [o for o in orders if o.side == "sell"]
    grid = np.arange(cfg.initial_price - 6, cfg.initial_price + 6, cfg.tick_size)
    demand = np.array([sum(o.size for o in buys if o.price >= p) for p in grid])
    supply = np.array([sum(o.size for o in sells if o.price <= p) for p in grid])
    p_star, volume = ex._find_clearing_price(buys, sells)

    fig, ax = _fig(1, 2, 2.5)
    ax[0].step(grid, demand, where="post", color="#1f77b4", label="спрос: объём с ценой ≥ p")
    ax[0].step(grid, supply, where="post", color="#d62728", label="предложение: объём с ценой ≤ p")
    ax[0].plot(grid, np.minimum(demand, supply), color="black", lw=1.4,
               label="исполняемый объём min(спрос, предложение)")
    if p_star is not None:
        ax[0].axvline(p_star, color="green", ls="--", lw=1.0)
        ax[0].plot([p_star], [volume], "o", color="green", ms=5)
        ax[0].annotate(f"p* = {p_star:.2f}\nобъём {volume:g}",
                       xy=(p_star, volume), xytext=(6, -18),
                       textcoords="offset points", color="green")
    ax[0].set_title("а) Кривые спроса и предложения в тике")
    ax[0].set_xlabel("цена")
    ax[0].set_ylabel("объём")
    ax[0].legend(loc="upper center", fontsize=6.5)

    # что делает сдвиг якоря: та же механика, центр генерации смещён
    shifts = np.linspace(-2.0, 2.0, 21)
    prices, volumes = [], []
    for shift in shifts:
        e = LimitExchange(cfg.venue1, cfg.tick_size, cfg.initial_price,
                          np.random.default_rng(11))
        found, vol = [], []
        for trial in range(40):
            e.rng = np.random.default_rng(1000 + trial)
            os_ = e.generate_background_orders(cfg.initial_price + shift, 1)
            p, v = e._find_clearing_price([o for o in os_ if o.side == "buy"],
                                          [o for o in os_ if o.side == "sell"])
            if p is not None:
                found.append(p)
                vol.append(v)
        prices.append(np.mean(found) if found else np.nan)
        volumes.append(np.mean(vol) if vol else np.nan)
    ax[1].plot(shifts, np.array(prices) - cfg.initial_price, color="#1f77b4",
               marker="o", ms=2.5, label="средняя цена аукциона − 100")
    ax[1].plot(shifts, shifts, color="black", ls=":", label="сдвиг якоря (идеал)")
    twin = ax[1].twinx()
    twin.plot(shifts, volumes, color="#d62728", lw=0.9, label="объём")
    twin.set_ylim(0, max(volumes) * 1.6)
    twin.set_ylabel("объём аукциона", color="#d62728")
    twin.grid(False)
    ax[1].set_title("б) Аукцион переносит сдвиг якоря в цену")
    ax[1].set_xlabel("сдвиг якоря относительно 100")
    ax[1].set_ylabel("сдвиг цены аукциона")
    ax[1].legend(loc="upper left")
    return fig


def ex_book_formation(run: D.RunData, ph: D.Phases):
    """Как из случайного потока заявок получаются стакан, спред и глубина."""
    cfg = run.cfg
    x = np.linspace(-3 * cfg.venue1.price_std, 3 * cfg.venue1.price_std, 400)
    density = np.exp(-0.5 * (x / cfg.venue1.price_std) ** 2)
    density /= density.max()

    fig, ax = _fig(1, 3, 2.4)
    ax[0].plot(x, density, color="#1f77b4", label="цены покупок")
    ax[0].plot(x, density, color="#d62728", ls="--", label="цены продаж")
    ax[0].fill_between(x, density, color="#999999", alpha=0.3)
    ax[0].axvline(0, color="black", lw=0.8, ls=":")
    ax[0].set_ylim(0, 1.45)
    ax[0].annotate("сторона и цена независимы,\nпоэтому в половине пар\n"
                   "покупка оказывается выше продажи",
                   xy=(0.0, 0.55), xytext=(-6.6, 1.05), fontsize=6,
                   arrowprops={"arrowstyle": "->", "lw": 0.7})
    ax[0].set_title("а) Один закон цен на обе стороны")
    ax[0].set_xlabel(f"цена заявки − якорь, N(0, {cfg.venue1.price_std:g})")
    ax[0].set_ylabel("плотность")
    ax[0].legend(loc="upper right", fontsize=6.5)

    # средний профиль реального стакана из записи прогона
    centers = 0.5 * (run.book_edges[1:] + run.book_edges[:-1])
    for v, name in enumerate(D.VENUES):
        bid = run.book_bid[:, v].mean(axis=0)
        ask = run.book_ask[:, v].mean(axis=0)
        ax[1].plot(centers, bid, color=VENUE_COLOR[v], lw=1.0,
                   label=f"{D.VENUE_LABEL[name]}: покупки")
        ax[1].plot(centers, ask, color=VENUE_COLOR[v], lw=1.0, ls="--",
                   label=f"{D.VENUE_LABEL[name]}: продажи")
    ax[1].axvline(0, color="black", lw=0.8)
    ax[1].set_title("б) Профиль стакана")
    ax[1].set_xlabel("цена − мид")
    ax[1].set_ylabel("средний объём в бине")
    ax[1].legend(loc="upper left", fontsize=5.5)

    # мини-эксперимент: чем гуще поток, тем уже спред и больше глубина
    rates = [1.0, 2.0, 4.0, 7.0, 10.0, 15.0]
    spreads, depths = [], []
    for rate in rates:
        cfg_a = ExchangeConfig(name="1", arrival_rate=rate,
                               order_size=cfg.venue1.order_size,
                               order_ttl=cfg.venue1.order_ttl,
                               price_std=cfg.venue1.price_std,
                               ewma_half_life=cfg.venue1.ewma_half_life)
        market = CoupledMarket(cfg_a, cfg.venue2, tick_size=cfg.tick_size,
                               initial_price=cfg.initial_price,
                               anchor_ewma_half_life=cfg.anchor_half_life,
                               depth_band=cfg.depth_band, seed=5)
        market.warmup(150)
        s, d = [], []
        for _ in range(200):
            market.step()
            ex = market.exchanges["1"]
            s.append(ex.spread if ex.spread is not None else np.nan)
            d.append(ex.depth_near_mid(cfg.depth_band))
        spreads.append(np.nanmean(s))
        depths.append(np.mean(d))
    ax[2].plot(rates, spreads, color="#d62728", marker="o", ms=3,
               label="средний спред")
    ax[2].set_xlabel("интенсивность потока заявок, шт/тик")
    ax[2].set_ylabel("спред", color="#d62728")
    twin = ax[2].twinx()
    twin.plot(rates, depths, color="#1f77b4", marker="s", ms=3,
              label="глубина у мида")
    twin.set_ylabel("глубина", color="#1f77b4")
    twin.grid(False)
    ax[2].axvline(cfg.venue1.arrival_rate, color="black", ls=":", lw=0.8)
    ax[2].axvline(cfg.venue2.arrival_rate, color="black", ls=":", lw=0.8)
    ax[2].annotate("биржи 2 и 1\nв прогоне", xy=(cfg.venue2.arrival_rate, spreads[1]),
                   xytext=(4.5, max(spreads) * 0.75), fontsize=6.5)
    ax[2].set_title("в) Поток → спред и глубина")
    return fig


def ex_coupling(run: D.RunData, ph: D.Phases):
    """Связь площадок: общий якорь и вес по глубине."""
    cfg = run.cfg
    market = CoupledMarket(cfg.venue1, cfg.venue2, tick_size=cfg.tick_size,
                           initial_price=cfg.initial_price,
                           anchor_ewma_half_life=cfg.anchor_half_life,
                           depth_band=cfg.depth_band, seed=17)
    market.warmup(300)
    hist = {"t": [], "m1": [], "m2": [], "a": []}
    for t in range(-60, 0):
        market.step()
        hist["t"].append(t)
        hist["m1"].append(market.exchanges["1"].mid)
        hist["m2"].append(market.exchanges["2"].mid)
        hist["a"].append(market.anchor)
    market.anchor *= 1.02                       # разовый шок якоря +2%
    for t in range(0, 200):
        market.step()
        hist["t"].append(t)
        hist["m1"].append(market.exchanges["1"].mid)
        hist["m2"].append(market.exchanges["2"].mid)
        hist["a"].append(market.anchor)

    fig, ax = _fig(1, 2, 2.5)
    ax[0].plot(hist["t"], hist["m1"], color=VENUE_COLOR[0], label="мид биржи 1 (толстая)")
    ax[0].plot(hist["t"], hist["m2"], color=VENUE_COLOR[1], label="мид биржи 2 (тонкая)")
    ax[0].plot(hist["t"], hist["a"], color=ANCHOR_COLOR, ls="--", label="якорь")
    ax[0].axvline(0, color="black", lw=0.8)
    ax[0].annotate("разовый шок якоря +2%", xy=(0, hist["a"][60]),
                   xytext=(15, hist["a"][60] - 1.6), fontsize=7,
                   arrowprops={"arrowstyle": "->", "lw": 0.7})
    ax[0].set_title("а) Отклик обеих площадок на шок якоря")
    ax[0].set_xlabel("тик от шока")
    ax[0].set_ylabel("цена")
    ax[0].legend(loc="lower right")

    w1 = run.depth[:, 0] / np.maximum(run.depth.sum(axis=1), 1e-9)
    ax[1].plot(D.moving_average(w1, _smooth_window(run)), color=VENUE_COLOR[0],
               label="вес биржи 1 в якоре")
    ax[1].axhline(0.5, color="black", ls=":", lw=0.8, label="равные веса")
    ax[1].set_ylim(0, 1)
    ax[1].set_title("б) Вес площадки в якоре = её доля глубины")
    ax[1].set_xlabel("тик")
    ax[1].set_ylabel("доля")
    ax[1].legend(loc="lower right")
    _thousands(ax[1])
    return fig


def ex_ce_clearing(run: D.RunData, ph: D.Phases):
    """Клиринг непрерывной биржи: взвешенный МНК в логарифмах цен."""
    ratios = np.logspace(-1.5, 1.5, 41)
    prices = []
    for ratio in ratios:
        ex = PortfolioExchange(assets=["X", "Y"], prices={"X": 1.0, "Y": 100.0},
                               unit_of_account="X")
        ex.submit(Order(weights={"Y": +1, "X": -1}, z=105.0, lam=1.0 * ratio,
                        agent="покупатель"))
        ex.submit(Order(weights={"Y": +1, "X": -1}, z=95.0, lam=1.0,
                        agent="продавец"))
        rep = ex.step(dt=1.0)
        prices.append(rep.prices["Y"])

    fig, ax = _fig(1, 2, 2.5)
    ax[0].semilogx(ratios, prices, color=CE_COLOR, marker="o", ms=2.5,
                   label="цена клиринга")
    ax[0].axhline(np.sqrt(105 * 95), color="black", ls="--", lw=0.8,
                  label=f"√(105·95) = {np.sqrt(105 * 95):.2f} при равных λ")
    ax[0].axhline(105, color="#1f77b4", ls=":", lw=0.8, label="z покупателя = 105")
    ax[0].axhline(95, color="#d62728", ls=":", lw=0.8, label="z продавца = 95")
    ax[0].set_title("а) Цена = λ-взвешенное среднее котировок")
    ax[0].set_xlabel("λ покупателя / λ продавца")
    ax[0].set_ylabel("цена Y после клиринга")
    ax[0].legend(loc="upper left", fontsize=6.5)

    grid = np.linspace(90, 110, 200)
    f_buy = np.log(105.0) - np.log(grid)
    f_sell = np.log(95.0) - np.log(grid)
    ax[1].plot(grid, f_buy, color="#1f77b4", label="f покупателя = ln(z) − ln(P)")
    ax[1].plot(grid, f_sell, color="#d62728", label="f продавца")
    ax[1].plot(grid, f_buy + f_sell, color="black", lw=1.3,
               label="сумма потоков (обнуляется в цене клиринга)")
    ax[1].axhline(0, color="black", lw=0.6)
    ax[1].axvline(np.sqrt(105 * 95), color=CE_COLOR, ls="--", lw=1.0)
    ax[1].set_title("б) Клиринг = обнуление суммарного потока стоимости")
    ax[1].set_xlabel("цена Y")
    ax[1].set_ylabel("лог-расхождение f")
    ax[1].legend(loc="upper right", fontsize=6.5)
    return fig


def ex_shading(run: D.RunData, ph: D.Phases):
    """Шейдинг котировки: позиция сдвигает цену безразличия против себя."""
    cfg = run.cfg
    mid = float(run.mid[ph.stationary[1], 0])
    sigma2 = float(np.mean(run.vol[D.window_slice(ph.stationary), 0]) ** 2)
    cap = cfg.c0
    floor = cfg.capital_floor_frac * cfg.c0
    q = np.linspace(-30, 30, 300)

    fig, ax = _fig(1, 2, 2.5)
    for h_r, color in zip(cfg.h_r_values, plt.cm.viridis(np.linspace(0, 0.85, 4))):
        g = F.risk_coefficient(h_r, run.gamma, sigma2, cap, floor)
        z = np.array([F.shaded_quote(mid, g, qi, cfg.shading_clamp) for qi in q])
        ax[0].plot(q, z - mid, color=color, label=f"h_R = {h_r:g}")
    ax[0].axhline(0, color="black", lw=0.6)
    ax[0].axvline(0, color="black", lw=0.6)
    ax[0].set_title("а) Сдвиг котировки z − мид от стоимости позиции")
    ax[0].set_xlabel("стоимость позиции q, X1")
    ax[0].set_ylabel("z − мид")
    ax[0].legend(loc="upper right")

    g = F.risk_coefficient(1.0, run.gamma, sigma2, cap, floor)
    depth = float(np.mean(run.depth[D.window_slice(ph.stationary), 0]))
    spread = float(np.nanmean(run.spread[D.window_slice(ph.stationary), 0]))
    H = F.book_quality(depth, mid, spread, cfg.tick_size)
    for h_m, color in zip(cfg.h_m_values, plt.cm.plasma(np.linspace(0, 0.8, 4))):
        lam = F.lam_translator(cfg.kappa_t, H, 1.0 / 16.0, h_m)
        z = np.array([F.shaded_quote(mid, g, qi, cfg.shading_clamp) for qi in q])
        flow = lam * (np.log(z) - np.log(mid))
        ax[1].plot(q, flow, color=color, label=f"h_m = {h_m:g}, λ = {lam:.1f}")
    ax[1].axhline(0, color="black", lw=0.6)
    ax[1].set_title("б) Возвращающий поток λ·f при цене CE, равной миду")
    ax[1].set_xlabel("стоимость позиции q, X1")
    ax[1].set_ylabel("нотионал за тик, X1")
    ax[1].legend(loc="upper right")
    return fig


def ex_hedge(run: D.RunData, ph: D.Phases):
    """Хедж-порог: риск последней единицы позиции против потери на сбросе."""
    cfg = run.cfg
    sl = D.window_slice(ph.stationary)
    mid = float(np.mean(run.mid[sl, 0]))
    spread = float(np.nanmean(run.spread[sl, 0]))
    sigma2 = float(np.mean(run.vol[sl, 0]) ** 2)
    cost = F.hedge_cost(spread, mid, cfg.tick_size)
    floor = cfg.capital_floor_frac * cfg.c0
    q = np.linspace(0, 40, 300)

    fig, ax = _fig(1, 2, 2.5)
    ax[0].axhline(cost * 1e4, color="black", ls="--",
                  label=f"потеря на хедже cost = ½·спред/мид = {cost * 1e4:.1f} б.п.")
    for h_r, color in zip(cfg.h_r_values, plt.cm.viridis(np.linspace(0, 0.85, 4))):
        g = F.risk_coefficient(h_r, run.gamma, sigma2, cfg.c0, floor)
        ax[0].plot(q, g * q * 1e4, color=color, label=f"риск g·|q|, h_R = {h_r:g}")
        q_star = cost / g
        ax[0].plot([q_star], [cost * 1e4], "o", color=color, ms=4)
    ax[0].set_title("а) Порог хеджа — точка равенства риска и потери")
    ax[0].set_xlabel("стоимость позиции |q|, X1")
    ax[0].set_ylabel("в базисных пунктах")
    ax[0].legend(loc="upper left", fontsize=6.5)

    # игрушечная траектория инвентаря: приток случаен, хедж срезает излишек
    rng = np.random.default_rng(4)
    flow = rng.normal(0.0, 1.2, 400)
    for h_r, color in zip((0.55, 1.35), ("#1f77b4", "#d62728")):
        g = F.risk_coefficient(h_r, run.gamma, sigma2, cfg.c0, floor)
        q_star = cost / g
        pos, path = 0.0, []
        for step in flow:
            pos += step
            excess = F.hedge_excess_value(g, pos, cost)
            if excess > 0:
                pos -= np.sign(pos) * excess
            path.append(pos)
        ax[1].plot(path, color=color, lw=0.9, label=f"h_R = {h_r:g}, порог {q_star:.1f}")
        ax[1].axhline(q_star, color=color, ls=":", lw=0.8)
        ax[1].axhline(-q_star, color=color, ls=":", lw=0.8)
    ax[1].plot(np.cumsum(flow), color="#999999", lw=0.9, ls="--",
               label="без хеджа: позиция блуждает")
    ax[1].set_title("б) Хедж удерживает инвентарь в коридоре ±cost/g")
    ax[1].set_xlabel("тик")
    ax[1].set_ylabel("стоимость позиции, X1")
    ax[1].legend(loc="upper left", fontsize=6.5)
    return fig


# --------------------------------------------------------------------------- #
# МИР
# --------------------------------------------------------------------------- #

def fig_world_price(run: D.RunData, ph: D.Phases):
    """Справедливая цена, якорь и цены обеих площадок."""
    T = run.T
    t = np.arange(T + 1)
    win = _smooth_window(run)

    fig, ax = _fig(2, 2, 4.3)
    ax[0].plot(t, run.fundamental, color=FUND_COLOR, lw=1.2, label="справедливая цена")
    ax[0].plot(t, run.anchor, color=ANCHOR_COLOR, lw=0.9, ls="--", label="якорь")
    for v, name in enumerate(D.VENUES):
        ax[0].plot(t, run.mid[:, v], color=VENUE_COLOR[v], lw=0.7, alpha=0.75,
                   label=f"мид, {D.VENUE_LABEL[name]}")
    _phase_spans(ax[0], ph)
    ax[0].set_title("а) Справедливая цена, якорь и миды")
    ax[0].set_xlabel("тик")
    ax[0].set_ylabel("цена")
    ax[0].legend(loc="upper left")
    _thousands(ax[0])

    lo = max(ph.stationary[1] - 400, 0)
    seg = slice(lo, ph.stationary[1] + 1)
    ax[1].plot(t[seg], run.fundamental[seg], color=FUND_COLOR, lw=1.2,
               label="справедливая цена")
    for v, name in enumerate(D.VENUES):
        ax[1].plot(t[seg], run.mid[seg, v], color=VENUE_COLOR[v], lw=0.8,
                   label=f"мид, биржа {name}")
        ax[1].fill_between(t[seg], run.best_bid[seg, v], run.best_ask[seg, v],
                           color=VENUE_COLOR[v], alpha=0.18, lw=0)
    ax[1].set_title("б) Фрагмент: миды и полосы котировок")
    ax[1].set_xlabel("тик")
    ax[1].set_ylabel("цена")
    ax[1].legend(loc="upper left")
    _thousands(ax[1])

    dev = (run.mid / run.fundamental[:, None] - 1.0) * 1e4
    for v, name in enumerate(D.VENUES):
        ax[2].plot(t, D.moving_average(dev[:, v], win), color=VENUE_COLOR[v],
                   label=f"биржа {name}")
    ax[2].axhline(0, color="black", lw=0.7)
    _phase_spans(ax[2], ph)
    ax[2].set_title("в) Отклонение мида от справедливой цены")
    ax[2].set_xlabel("тик")
    ax[2].set_ylabel(f"б.п., среднее за {win} тиков")
    ax[2].legend(loc="upper left")
    _thousands(ax[2])

    for v, name in enumerate(D.VENUES):
        ax[3].hist(dev[1:, v], bins=80, histtype="step", density=True,
                   color=VENUE_COLOR[v],
                   label=f"биржа {name}: σ = {np.std(dev[1:, v]):.0f} б.п.")
    ax[3].axvline(0, color="black", lw=0.7)
    ax[3].set_title("г) Распределение отклонений")
    ax[3].set_xlabel("базисные пункты")
    ax[3].set_ylabel("плотность")
    ax[3].legend(loc="upper left")
    return fig


def fig_world_book(run: D.RunData, ph: D.Phases):
    """Поведение стаканов: тепловые карты, средний профиль, спред."""
    centers = 0.5 * (run.book_edges[1:] + run.book_edges[:-1])
    fig, ax = _fig(2, 2, 4.3)

    for v, name in enumerate(D.VENUES):
        net = run.book_bid[:, v] - run.book_ask[:, v]
        scale = np.percentile(np.abs(net), 99) or 1.0
        mesh = ax[v].pcolormesh(run.book_ticks, centers, net.T, cmap="RdBu",
                                vmin=-scale, vmax=scale, shading="nearest",
                                rasterized=True)
        ax[v].axhline(0, color="black", lw=0.6)
        ax[v].set_title(f"{'аб'[v]}) Стакан во времени, {D.VENUE_LABEL[name]}")
        ax[v].set_xlabel("тик")
        ax[v].set_ylabel("цена − мид")
        bar = fig.colorbar(mesh, ax=ax[v], pad=0.01)
        bar.set_label("покупки − продажи", fontsize=6.5)
        _thousands(ax[v])

    for v, name in enumerate(D.VENUES):
        bid = run.book_bid[:, v].mean(axis=0)
        ask = run.book_ask[:, v].mean(axis=0)
        ax[2].fill_between(centers, bid, color=VENUE_COLOR[v], alpha=0.25, lw=0)
        ax[2].plot(centers, bid, color=VENUE_COLOR[v], lw=1.0,
                   label=f"биржа {name}: покупки")
        ax[2].plot(centers, ask, color=VENUE_COLOR[v], lw=1.0, ls="--",
                   label=f"биржа {name}: продажи")
    ax[2].axvline(0, color="black", lw=0.7)
    ax[2].set_title("в) Средний профиль стакана")
    ax[2].set_xlabel("цена − мид")
    ax[2].set_ylabel("средний объём в бине")
    ax[2].legend(loc="upper left", fontsize=6.5)

    win = _smooth_window(run)
    top = 0.0
    for v, name in enumerate(D.VENUES):
        share = np.mean(np.isnan(run.spread[1:, v])) * 100
        smooth = D.moving_average(run.spread[:, v], win)
        top = max(top, np.nanmax(smooth))
        ax[3].plot(smooth, color=VENUE_COLOR[v],
                   label=f"биржа {name}: односторонний стакан {share:.1f}% тиков")
    ax[3].axhline(run.cfg.tick_size, color="black", ls=":", lw=0.8,
                  label="один тик цены")
    _phase_spans(ax[3], ph)
    ax[3].set_ylim(0, top * 1.35)
    ax[3].set_title("г) Спред площадок")
    ax[3].set_xlabel("тик")
    ax[3].set_ylabel(f"спред, среднее за {win} тиков")
    ax[3].legend(loc="upper right", fontsize=6.5)
    _thousands(ax[3])
    return fig


def fig_world_volume(run: D.RunData, ph: D.Phases):
    """Объёмы торгов на лимитных биржах и вклад хеджей."""
    win = _smooth_window(run)
    fig, ax = _fig(2, 2, 4.3)

    for v, name in enumerate(D.VENUES):
        ax[0].plot(D.moving_average(run.volume[:, v], win), color=VENUE_COLOR[v],
                   label=f"биржа {name}")
    _phase_spans(ax[0], ph)
    ax[0].set_title("а) Объём аукциона за тик")
    ax[0].set_xlabel("тик")
    ax[0].set_ylabel(f"объём, шт (среднее за {win} тиков)")
    ax[0].legend(loc="upper left")
    _thousands(ax[0])

    for v, name in enumerate(D.VENUES):
        ax[1].plot(np.cumsum(run.volume[:, v]), color=VENUE_COLOR[v],
                   label=f"биржа {name}: {run.volume[:, v].sum():,.0f} шт"
                         .replace(",", " "))
    ax[1].set_title("б) Накопленный объём")
    ax[1].set_xlabel("тик")
    ax[1].set_ylabel("объём, шт")
    ax[1].legend(loc="upper left")
    _thousands(ax[1])

    bins = np.arange(0, np.percentile(run.volume[1:], 99.5) + 1.0, 1.0)
    for v, name in enumerate(D.VENUES):
        idle = np.mean(run.volume[1:, v] == 0) * 100
        ax[2].hist(run.volume[1:, v], bins=bins, histtype="step", density=True,
                   color=VENUE_COLOR[v],
                   label=f"биржа {name}: без сделок {idle:.1f}% тиков")
    ax[2].set_title("в) Распределение объёма аукциона")
    ax[2].set_xlabel("объём за тик, шт")
    ax[2].set_ylabel("плотность")
    ax[2].legend(loc="upper right")

    for v, name in enumerate(D.VENUES):
        hedge_units = run.hedge_notional[:, v] / np.maximum(run.mid[:, v], 1e-9)
        share = 100 * D.moving_average(hedge_units, win * 5) / np.maximum(
            D.moving_average(run.volume[:, v], win * 5), 1e-9)
        ax[3].plot(share, color=VENUE_COLOR[v],
                   label=f"биржа {name}: всего {run.hedge_count[:, v].sum():.0f} хеджей")
    _phase_spans(ax[3], ph)
    ax[3].set_title("г) Хеджи трансляторов как доля объёма площадки")
    ax[3].set_xlabel("тик")
    ax[3].set_ylabel("% объёма аукционов")
    ax[3].legend(loc="upper right")
    _thousands(ax[3])
    return fig


def fig_world_liquidity(run: D.RunData, ph: D.Phases):
    """Ликвидность: глубина, качество книги, цена исполнения, волатильность."""
    cfg = run.cfg
    win = _smooth_window(run)
    fig, ax = _fig(2, 2, 4.3)

    for v, name in enumerate(D.VENUES):
        ax[0].plot(D.moving_average(run.depth[:, v], win), color=VENUE_COLOR[v],
                   label=f"биржа {name}")
    _phase_spans(ax[0], ph)
    ax[0].set_title(f"а) Глубина в полосе ±{cfg.depth_band:g} от мида")
    ax[0].set_xlabel("тик")
    ax[0].set_ylabel("объём, шт")
    ax[0].legend(loc="upper left")
    _thousands(ax[0])

    for v, name in enumerate(D.VENUES):
        H = np.array([F.book_quality(d, m, s, cfg.tick_size)
                      for d, m, s in zip(run.depth[:, v], run.mid[:, v],
                                         run.spread[:, v])])
        ax[1].plot(D.moving_average(H, win), color=VENUE_COLOR[v],
                   label=f"H, биржа {name}")
    ax[1].set_title("б) Качество книги H = глубина·мид·тик / спред")
    ax[1].set_xlabel("тик")
    ax[1].set_ylabel("H, X1")
    secondary = ax[1].secondary_yaxis(
        "right", functions=(lambda h: h * cfg.kappa_t, lambda l: l / cfg.kappa_t))
    secondary.set_ylabel(f"мощность семейства Λ = {cfg.kappa_t:g}·H")
    ax[1].legend(loc="upper left")
    _thousands(ax[1])

    centers = 0.5 * (run.book_edges[1:] + run.book_edges[:-1])
    sizes = np.arange(1.0, 41.0)
    for v, name in enumerate(D.VENUES):
        ask = run.book_ask[:, v]
        cum = np.cumsum(ask, axis=1)
        cost_value = np.cumsum(ask * centers[None, :], axis=1)
        avg = np.full((ask.shape[0], sizes.size), np.nan)
        for s, size in enumerate(sizes):
            idx = np.argmax(cum >= size, axis=1)
            ok = cum[np.arange(ask.shape[0]), idx] >= size
            over = cum[np.arange(ask.shape[0]), idx] - size
            value = (cost_value[np.arange(ask.shape[0]), idx]
                     - over * centers[idx])
            avg[ok, s] = value[ok] / size
        # кривая обрывается там, где видимой глубины хватает уже не всегда:
        # усреднять по одним лишь «удачным» срезам значило бы приукрасить книгу
        filled = np.mean(~np.isnan(avg), axis=0)
        curve = np.full(sizes.size, np.nan)
        usable = filled >= 0.8
        if usable.any():
            curve[usable] = np.nanmean(avg[:, usable], axis=0)
        last = np.flatnonzero(~np.isnan(curve))
        ax[2].plot(sizes, curve, color=VENUE_COLOR[v],
                   label=f"биржа {name}: глубины хватает до "
                         f"{sizes[last[-1]]:.0f} шт")
        if last.size:
            ax[2].plot([sizes[last[-1]]], [curve[last[-1]]], "o",
                       color=VENUE_COLOR[v], ms=4)
    ax[2].set_title("в) Цена немедленной покупки по книге")
    ax[2].set_xlabel("размер заявки, шт")
    ax[2].set_ylabel("средняя цена − мид")
    ax[2].legend(loc="upper left", fontsize=6.5)

    for v, name in enumerate(D.VENUES):
        ax[3].plot(D.moving_average(run.vol[:, v], win) * 1e4,
                   color=VENUE_COLOR[v], label=f"биржа {name}")
    ax[3].axhline(cfg.fundamental_vol * 1e4, color=FUND_COLOR, ls="--", lw=0.9,
                  label="волатильность справедливой цены")
    _phase_spans(ax[3], ph)
    ax[3].set_title("г) EWMA-волатильность мида за тик")
    ax[3].set_xlabel("тик")
    ax[3].set_ylabel("б.п. за тик")
    ax[3].legend(loc="upper right")
    _thousands(ax[3])
    return fig


# --------------------------------------------------------------------------- #
# НЕПРЕРЫВНАЯ БИРЖА
# --------------------------------------------------------------------------- #

def fig_ce_price(run: D.RunData, ph: D.Phases):
    """Цена CE против мидов и справедливой цены: прогрев и стационар."""
    t = np.arange(run.T + 1)
    err = D.translation_error_bp(run)
    win = _smooth_window(run)
    y1 = run.price("Y1")
    y2 = run.price("Y2") / run.price("X2")

    fig, ax = _fig(2, 2, 4.3)
    for k, (label, rng_) in enumerate((("прогрев", ph.warmup),
                                       ("стационар", ph.stationary))):
        lo = rng_[0] if k == 0 else max(rng_[1] - 400, rng_[0])
        hi = min(lo + 400, rng_[1])
        seg = slice(lo, hi + 1)
        ax[k].plot(t[seg], run.fundamental[seg], color=FUND_COLOR, lw=1.1,
                   label="справедливая цена")
        ax[k].plot(t[seg], run.mid[seg, 0], color=VENUE_COLOR[0], lw=0.8,
                   label="мид биржи 1")
        ax[k].plot(t[seg], y1[seg], color=CE_COLOR, lw=1.0, label="цена Y1 на CE")
        ax[k].plot(t[seg], y2[seg], color="#9467bd", lw=0.8, ls="--",
                   label="цена Y2 на CE (в X2)")
        ax[k].set_title(f"{'аб'[k]}) {label}: тики {lo}–{hi}")
        ax[k].set_xlabel("тик")
        ax[k].set_ylabel("цена")
        ax[k].legend(loc="upper left", fontsize=6.5)
        _thousands(ax[k])

    for v, name in enumerate(D.VENUES):
        ax[2].plot(D.moving_average(err[:, v], win), color=VENUE_COLOR[v],
                   label=f"Y{name} на CE против мида биржи {name}")
    ax[2].axhline(0, color="black", lw=0.7)
    _phase_spans(ax[2], ph)
    ax[2].set_title("в) Ошибка трансляции цены на CE")
    ax[2].set_xlabel("тик")
    ax[2].set_ylabel(f"б.п., среднее за {win} тиков")
    ax[2].legend(loc="upper left", fontsize=6.5)
    _thousands(ax[2])

    data, labels, colors = [], [], []
    for label, rng_ in ph.named:
        sl = D.window_slice(rng_)
        for v, name in enumerate(D.VENUES):
            values = err[sl, v]
            data.append(values[~np.isnan(values)])
            labels.append(f"{SHORT_PHASE[label]}\nY{name}")
            colors.append(VENUE_COLOR[v])
    _box(ax[3], data, labels, colors)
    ax[3].axhline(0, color="black", lw=0.7)
    ax[3].tick_params(axis="x", labelsize=6)
    ax[3].set_title("г) Ошибка трансляции по фазам")
    ax[3].set_ylabel("базисные пункты")
    return fig


def fig_ce_basis(run: D.RunData, ph: D.Phases):
    """Базисы арбитражёров и расхождения, которые видят заявки."""
    bas = D.basis_bp(run)
    win = _smooth_window(run)
    fig, ax = _fig(2, 2, 4.3)

    ax[0].plot(run.price("X2"), color="#9467bd", lw=0.7,
               label="цена X2 в единицах X1")
    ax[0].axhline(1.0, color="black", ls="--", lw=0.8, label="паритет площадок")
    _phase_spans(ax[0], ph)
    ax[0].set_title("а) Курс кассовых активов двух площадок")
    ax[0].set_xlabel("тик")
    ax[0].set_ylabel("P(X2)")
    ax[0].legend(loc="upper left")
    _thousands(ax[0])

    span = 0.0
    for j, (label, color) in enumerate((("AX: X1 против X2", KIND_COLOR["AX"]),
                                        ("AY: Y1 против Y2", KIND_COLOR["AY"]))):
        smooth = D.moving_average(bas[:, j], win)
        # первые тики книга CE ещё пуста, стартовый выброс не характерен
        span = max(span, np.nanpercentile(np.abs(smooth[200:]), 99.5))
        ax[1].plot(smooth, color=color, label=label)
    ax[1].axhline(0, color="black", lw=0.7)
    _phase_spans(ax[1], ph)
    ax[1].set_ylim(-span * 1.6, span * 1.6)
    ax[1].set_title("б) Базисы арбитражёров")
    ax[1].set_xlabel("тик")
    ax[1].set_ylabel(f"б.п., среднее за {win} тиков")
    ax[1].legend(loc="upper left")
    _thousands(ax[1])

    data, labels, colors = [], [], []
    for label, rng_ in ph.named:
        sl = D.window_slice(rng_)
        for j, kind in enumerate(("AX", "AY")):
            data.append(bas[sl, j])
            labels.append(f"{SHORT_PHASE[label]}\n{kind}")
            colors.append(KIND_COLOR[kind])
    _box(ax[2], data, labels, colors)
    ax[2].axhline(0, color="black", lw=0.7)
    ax[2].tick_params(axis="x", labelsize=6)
    ax[2].set_title("в) Базисы по фазам")
    ax[2].set_ylabel("базисные пункты")

    for k, kind in enumerate(D.KINDS):
        ax[3].plot(D.moving_average(run.ce_absf_kind[:, k] * 1e4, win * 2),
                   color=KIND_COLOR[kind], label=D.KIND_LABEL[kind])
    _phase_spans(ax[3], ph)
    ax[3].set_yscale("log")
    ax[3].set_title("г) Среднее |f| — расхождение, которое видит заявка типа")
    ax[3].set_xlabel("тик")
    ax[3].set_ylabel("базисные пункты, лог. шкала")
    ax[3].legend(loc="upper right", fontsize=6.5)
    _thousands(ax[3])
    return fig


def fig_ce_clearing(run: D.RunData, ph: D.Phases):
    """Оборот непрерывной биржи и качество клиринга."""
    win = _smooth_window(run)
    fig, ax = _fig(2, 2, 4.3)

    ax[0].plot(D.moving_average(run.ce_gross, win), color=CE_COLOR,
               label="оборот CE")
    _phase_spans(ax[0], ph)
    ax[0].set_title(f"а) Оборот CE за тик (среднее за {win} тиков)")
    ax[0].set_xlabel("тик")
    ax[0].set_ylabel("нотионал, X1")
    ax[0].legend(loc="upper left")
    _thousands(ax[0])

    smooth = np.column_stack([D.moving_average(run.ce_gross_kind[:, k], win * 2)
                              for k in range(len(D.KINDS))])
    ax[1].stackplot(np.arange(run.T + 1), smooth.T,
                    colors=[KIND_COLOR[k] for k in D.KINDS],
                    labels=[D.KIND_LABEL[k] for k in D.KINDS], alpha=0.85)
    ax[1].set_title("б) Оборот в разрезе типов агентов")
    ax[1].set_xlabel("тик")
    ax[1].set_ylabel("нотионал, X1")
    ax[1].legend(loc="upper left", fontsize=6.5)
    _thousands(ax[1])

    ax[2].plot(D.moving_average(run.ce_orders, win), color="#9467bd",
               label="исполнившихся заявок за тик")
    ax[2].set_xlabel("тик")
    ax[2].set_ylabel("шт")
    twin = ax[2].twinx()
    twin.plot(D.moving_average(run.ce_rank.astype(float), win), color="black",
              lw=0.9, label="ранг книги")
    twin.set_ylabel("ранг книги (из 4 активов)")
    twin.set_ylim(0, 4.2)
    twin.grid(False)
    ax[2].set_title("в) Наполнение книги CE")
    handles = [Line2D([], [], color="#9467bd", label="заявок за тик"),
               Line2D([], [], color="black", label="ранг книги")]
    ax[2].legend(handles=handles, loc="lower right", fontsize=6.5)
    _thousands(ax[2])

    ax[3].semilogy(np.maximum(run.ce_imbalance, 1e-18), color="#d62728", lw=0.5,
                   label="невязка клиринга, X1")
    ax[3].semilogy(np.maximum(run.pnl_residual, 1e-18), color="#1f77b4", lw=0.5,
                   label="невязка разложения PnL, X1")
    ax[3].axhline(np.median(run.ce_gross[1:]), color="black", ls="--", lw=0.8,
                  label="медианный оборот за тик")
    ax[3].set_title("г) Контроль точности: обе невязки — машинный ноль")
    ax[3].set_xlabel("тик")
    ax[3].set_ylabel("X1, лог. шкала")
    ax[3].legend(loc="center right", fontsize=6.5)
    _thousands(ax[3])
    return fig


# --------------------------------------------------------------------------- #
# АГЕНТЫ
# --------------------------------------------------------------------------- #

def fig_agents_pnl(run: D.RunData, ph: D.Phases):
    """PnL агентов и их копий: траектории, типы, фазы."""
    T = run.T
    fig, ax = _fig(2, 2, 4.3)

    for kind in D.KINDS:
        rows = run.kind_rows(kind)
        for i in rows:
            ax[0].plot(run.equity[i], color=KIND_COLOR[kind], lw=0.45, alpha=0.6)
    _phase_spans(ax[0], ph)
    ax[0].axhline(0, color="black", lw=0.7)
    handles = [Line2D([], [], color=KIND_COLOR[k], label=D.KIND_LABEL[k])
               for k in D.KINDS]
    ax[0].legend(handles=handles, loc="upper left", fontsize=6.5)
    ax[0].set_title("а) PnL каждой копии (64 траектории)")
    ax[0].set_xlabel("тик")
    ax[0].set_ylabel("PnL, X1")
    _thousands(ax[0])

    for kind in D.KINDS:
        rows = run.kind_rows(kind)
        ax[1].plot(run.equity[rows].sum(axis=0), color=KIND_COLOR[kind],
                   label=f"{kind}: итог {run.equity[rows, T].sum():+.3f}")
    ax[1].axhline(0, color="black", lw=0.7)
    _phase_spans(ax[1], ph)
    ax[1].set_title("б) Суммарный PnL по типам")
    ax[1].set_xlabel("тик")
    ax[1].set_ylabel("PnL, X1")
    ax[1].legend(loc="upper left", fontsize=6.5)
    _thousands(ax[1])

    order, colors, labels, values = [], [], [], []
    for kind in D.KINDS:
        rows = run.kind_rows(kind)
        rows = rows[np.argsort(-run.equity[rows, T])]
        for i in rows:
            order.append(i)
            values.append(run.equity[i, T])
            colors.append(KIND_COLOR[kind])
            labels.append(run.agents[i]["death_tick"] is not None)
    x = np.arange(len(order))
    bars = ax[2].bar(x, values, color=colors, width=0.85)
    for xi, bar, dead in zip(x, bars, labels):
        if dead:
            bar.set_alpha(0.35)
            bar.set_hatch("////")
    ax[2].axhline(0, color="black", lw=0.7)
    pos = 0
    for kind in D.KINDS:
        size = len(run.kind_rows(kind))
        ax[2].axvline(pos - 0.5, color="#cccccc", lw=0.7)
        ax[2].text(pos + size / 2, ax[2].get_ylim()[1], kind, ha="center",
                   va="top", fontsize=7)
        pos += size
    ax[2].set_xticks([])
    ax[2].set_title("в) Итоговый PnL по копиям")
    ax[2].set_xlabel("копии, внутри типа по убыванию PnL")
    ax[2].set_ylabel("PnL, X1")

    warm = D.phase_pnl(run, ph.warmup)
    stat = D.phase_pnl(run, ph.stationary)
    for kind in D.KINDS:
        rows = run.kind_rows(kind)
        alive = np.array([run.agents[i]["death_tick"] is None for i in rows])
        ax[3].scatter(warm[rows][alive], stat[rows][alive], s=16,
                      color=KIND_COLOR[kind])
        if (~alive).any():
            ax[3].scatter(warm[rows][~alive], stat[rows][~alive], s=16,
                          facecolors="none", edgecolors=KIND_COLOR[kind],
                          linewidths=0.8)
    ax[3].axhline(0, color="black", lw=0.7)
    ax[3].axvline(0, color="black", lw=0.7)
    ax[3].set_title("г) Прибыль: прогрев против стационара")
    ax[3].set_xlabel("PnL за прогрев, X1")
    ax[3].set_ylabel("PnL за стационар, X1")
    handles = [Line2D([], [], ls="", marker="o", ms=4, color=KIND_COLOR[k],
                      label=k) for k in D.KINDS]
    handles += [Line2D([], [], ls="", marker="o", ms=4, color="black",
                       label="выжил"),
                Line2D([], [], ls="", marker="o", ms=4, markerfacecolor="none",
                       color="black", label="выбыл")]
    ax[3].legend(handles=handles, loc="upper left", fontsize=6, ncol=3)
    return fig


def fig_agents_turnover(run: D.RunData, ph: D.Phases):
    """Объёмы торгов в разрезе агентов и их копий."""
    win = _smooth_window(run)
    fig, ax = _fig(2, 2, 4.3)

    # четыре линии ложатся друг на друга: условие нулевого потока стоимости
    # по каждому активу связывает нотионалы семейств в кольцо (см. текст),
    # поэтому рисуем их разной толщиной и штрихом — иначе видна только одна
    styles = [(2.6, "-"), (1.8, "-"), (1.1, "--"), (0.6, ":")]
    for k, kind in enumerate(D.KINDS):
        lw, ls = styles[k]
        ax[0].plot(D.moving_average(run.ce_gross_kind[:, k], win * 2),
                   color=KIND_COLOR[kind], lw=lw, ls=ls, label=kind)
    _phase_spans(ax[0], ph)
    ax[0].set_title("а) Оборот по типам агентов — совпадает")
    ax[0].set_xlabel("тик")
    ax[0].set_ylabel(f"X1 за тик, среднее за {win * 2} тиков")
    ax[0].legend(loc="lower left", fontsize=6.5, ncol=4)
    _thousands(ax[0])

    order, values, colors, dead = [], [], [], []
    for kind in D.KINDS:
        rows = run.kind_rows(kind)
        mean_turnover = run.turnover[rows].mean(axis=1)
        rows = rows[np.argsort(-mean_turnover)]
        for i in rows:
            order.append(i)
            values.append(run.turnover[i].mean())
            colors.append(KIND_COLOR[kind])
            dead.append(run.agents[i]["death_tick"] is not None)
    bars = ax[1].bar(np.arange(len(order)), values, color=colors, width=0.85)
    for bar, is_dead in zip(bars, dead):
        if is_dead:
            bar.set_alpha(0.35)
            bar.set_hatch("////")
    ax[1].set_xticks([])
    ax[1].set_title("б) Средний оборот копии за тик")
    ax[1].set_xlabel("копии по убыванию оборота")
    ax[1].set_ylabel("нотионал, X1")

    # берём прогрев: там живы все 64 копии, и зависимость от h_m видна целиком
    h_m = run.hyper("h_m")
    sl = D.window_slice(ph.warmup)
    for j, kind in enumerate(D.KINDS):
        rows = run.kind_rows(kind)
        values = run.turnover[rows][:, sl].mean(axis=1)
        ax[2].scatter(h_m[rows] * (1.0 + 0.02 * (j - 1.5)), values, s=16,
                      color=KIND_COLOR[kind], alpha=0.8, label=kind)
    reference = np.array(sorted(run.cfg.h_m_values), dtype=float)
    anchor_value = np.mean(run.turnover[run.kind_rows("AX")][:, sl].mean(axis=1))
    ax[2].plot(reference, anchor_value * np.mean(run.cfg.h_m_values) / reference,
               color="black", ls=":", lw=0.9, label="закон λ ∝ 1/h_m")
    ax[2].set_yscale("log")
    ax[2].set_xticks(list(run.cfg.h_m_values))
    ax[2].set_xticklabels([f"{v:g}" for v in run.cfg.h_m_values])
    ax[2].set_title("в) Оборот в прогреве против h_m")
    ax[2].set_xlabel("h_m (меньше — агрессивнее)")
    ax[2].set_ylabel("средний оборот за тик, X1")
    ax[2].legend(loc="lower left", fontsize=6.5, ncol=2)

    labels = [label for label, _ in ph.named]
    bottom = np.zeros(len(labels))
    for k, kind in enumerate(D.KINDS):
        shares = []
        for _, rng_ in ph.named:
            sl = D.window_slice(rng_)
            total = run.ce_gross_kind[sl].sum()
            shares.append(100 * run.ce_gross_kind[sl, k].sum() / max(total, 1e-12))
        shares = np.array(shares)
        ax[3].bar(labels, shares, bottom=bottom, color=KIND_COLOR[kind],
                  label=kind, width=0.6)
        for x, (share, base) in enumerate(zip(shares, bottom)):
            if share > 4:
                ax[3].text(x, base + share / 2, f"{share:.0f}%", ha="center",
                           va="center", fontsize=6.5, color="white")
        bottom += shares
    ax[3].set_title("г) Доля типа в обороте CE по фазам")
    ax[3].set_ylabel("% оборота фазы")
    ax[3].legend(loc="upper right", fontsize=6.5)
    return fig


def fig_agents_structure(run: D.RunData, ph: D.Phases):
    """Структурная аналитика PnL: переоценка позиции против результата хеджа."""
    fig, ax = _fig(2, 2, 4.3)

    for kind in ("T1", "T2"):
        rows = run.kind_rows(kind)
        ax[0].plot(np.cumsum(run.reval[rows].sum(axis=0)),
                   color=KIND_COLOR[kind], label=f"{kind}: переоценка")
        ax[0].plot(np.cumsum(run.hedge_pnl[rows].sum(axis=0)),
                   color=KIND_COLOR[kind], ls="--", label=f"{kind}: хедж")
        ax[0].plot(run.equity[rows].sum(axis=0), color=KIND_COLOR[kind], lw=1.6,
                   alpha=0.45, label=f"{kind}: итог")
    ax[0].axhline(0, color="black", lw=0.7)
    _phase_spans(ax[0], ph)
    ax[0].set_title("а) Трансляторы: из чего складывается PnL")
    ax[0].set_xlabel("тик")
    ax[0].set_ylabel("накопленным итогом, X1")
    ax[0].legend(loc="lower left", fontsize=6, ncol=2)
    _thousands(ax[0])

    x = np.arange(len(D.KINDS))
    reval = np.array([run.reval[run.kind_rows(k)].sum() for k in D.KINDS])
    hedge = np.array([run.hedge_pnl[run.kind_rows(k)].sum() for k in D.KINDS])
    ax[1].bar(x - 0.22, reval, width=0.2, color="#4c72b0", label="переоценка позиции")
    ax[1].bar(x, hedge, width=0.2, color="#dd8452", label="результат хеджа")
    ax[1].bar(x + 0.22, reval + hedge, width=0.2, color="#55a868", label="итог")
    ax[1].axhline(0, color="black", lw=0.7)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(D.KINDS)
    ax[1].set_title("б) Разложение итогового PnL по типам")
    ax[1].set_ylabel("X1")
    ax[1].legend(loc="upper left", fontsize=6.5)

    rows_t = np.concatenate([run.kind_rows("T1"), run.kind_rows("T2")])
    total = run.equity[rows_t, run.T]
    hedge_share = run.hedge_pnl[rows_t].sum(axis=1)
    reval_share = run.reval[rows_t].sum(axis=1)
    colors = [KIND_COLOR[run.agents[i]["kind"]] for i in rows_t]
    ax[2].scatter(reval_share, hedge_share, s=20, c=colors, alpha=0.8)
    lim = np.nanmax(np.abs(np.concatenate([reval_share, hedge_share]))) * 1.1
    ax[2].plot([-lim, lim], [lim, -lim], color="black", lw=0.7, ls=":",
               label="нулевой итог")
    ax[2].axhline(0, color="black", lw=0.6)
    ax[2].axvline(0, color="black", lw=0.6)
    ax[2].set_title("в) Трансляторы: переоценка против хеджа")
    ax[2].set_xlabel("переоценка позиции, X1")
    ax[2].set_ylabel("результат хеджа, X1")
    ax[2].legend(loc="upper right", fontsize=6.5)

    for kind in D.KINDS:
        rows = run.kind_rows(kind)
        inv = np.abs(run.inventory[rows]).mean(axis=1)
        ax[3].scatter(inv, run.equity[rows, run.T], s=18, color=KIND_COLOR[kind],
                      alpha=0.8, label=kind)
    ax[3].axhline(0, color="black", lw=0.7)
    ax[3].set_title("г) PnL против размера позиции")
    ax[3].set_xlabel("средняя |стоимость позиции|, X1")
    ax[3].set_ylabel("PnL, X1")
    ax[3].legend(loc="upper right", fontsize=6.5)
    return fig


def fig_agents_inventory(run: D.RunData, ph: D.Phases):
    """Инвентарь агентов и работа хеджа."""
    cfg = run.cfg
    win = _smooth_window(run)
    fig, ax = _fig(2, 2, 4.3)

    # трансляторы: позиция держится далеко ниже порога хеджа, поэтому обе
    # величины сравнимы только в логарифмическом масштабе
    for v, kind in enumerate(("T1", "T2")):
        rows = run.kind_rows(kind)
        inv = np.abs(run.inventory[rows])
        ax[0].plot(D.moving_average(np.median(inv, axis=0), win),
                   color=KIND_COLOR[kind], label=f"{kind}: медиана |позиции|")
        ax[0].plot(D.moving_average(np.percentile(inv, 90, axis=0), win),
                   color=KIND_COLOR[kind], ls=":", lw=0.9,
                   label=f"{kind}: 90-й процентиль")
        floor = cfg.capital_floor_frac * cfg.c0
        cost = 0.5 * np.maximum(run.spread[:, v], cfg.tick_size) / run.mid[:, v]
        g = run.gamma * run.vol[:, v] ** 2 / max(cfg.c0, floor)
        q_star = np.where(g > 0, cost / np.maximum(g, 1e-18), np.nan)
        ax[0].plot(q_star, color=KIND_COLOR[kind], lw=0.25, alpha=0.3)
        centers, med = _block_median(q_star, win)
        ax[0].plot(centers, med, color=KIND_COLOR[kind], ls="--", lw=1.1,
                   label=f"{kind}: порог хеджа cost/g при h_R = 1")
    ax[0].set_yscale("log")
    _phase_spans(ax[0], ph)
    ax[0].set_title("а) Трансляторы: |позиция| против порога хеджа")
    ax[0].set_xlabel("тик")
    ax[0].set_ylabel("X1, лог. шкала")
    ax[0].legend(loc="lower left", fontsize=5.5, ncol=2)
    _thousands(ax[0])

    for kind in ("AX", "AY"):
        rows = run.kind_rows(kind)
        inv = run.inventory[rows]
        ax[1].plot(D.moving_average(np.median(inv, axis=0), win),
                   color=KIND_COLOR[kind], label=f"{kind}: медиана")
        ax[1].fill_between(np.arange(run.T + 1),
                           D.moving_average(np.percentile(inv, 10, axis=0), win),
                           D.moving_average(np.percentile(inv, 90, axis=0), win),
                           color=KIND_COLOR[kind], alpha=0.2, lw=0)
    ax[1].axhline(0, color="black", lw=0.6)
    _phase_spans(ax[1], ph)
    ax[1].set_title("б) Арбитражёры: стоимость позиции")
    ax[1].set_xlabel("тик")
    ax[1].set_ylabel("X1")
    ax[1].legend(loc="upper left", fontsize=6.5)
    _thousands(ax[1])

    data, labels, colors = [], [], []
    for kind in D.KINDS:
        rows = run.kind_rows(kind)
        sl = D.window_slice(ph.stationary)
        data.append(np.abs(run.inventory[rows][:, sl]).ravel())
        labels.append(kind)
        colors.append(KIND_COLOR[kind])
    _box(ax[2], data, labels, colors)
    ax[2].set_yscale("log")
    ax[2].set_title("в) Распределение |позиции| в стационаре")
    ax[2].set_ylabel("X1, лог. шкала")

    for v, name in enumerate(D.VENUES):
        ax[3].plot(D.moving_average(run.hedge_count[:, v], win * 5) * 100,
                   color=VENUE_COLOR[v],
                   label=f"биржа {name}: {run.hedge_count[:, v].sum():.0f} хеджей, "
                         f"итог {run.hedge_result[:, v].sum():+.3f} X1")
    _phase_spans(ax[3], ph)
    ax[3].set_title("г) Частота хеджей трансляторов")
    ax[3].set_xlabel("тик")
    ax[3].set_ylabel("хеджей на 100 тиков")
    ax[3].legend(loc="upper right", fontsize=6.5)
    _thousands(ax[3])
    return fig


def fig_agents_hyper(run: D.RunData, ph: D.Phases):
    """Доли гиперпараметров в живых копиях во времени."""
    fig, ax = _fig(2, 2, 4.3)
    t = np.arange(run.T + 1)

    panels = [(0, ("T1", "T2"), "h_m", "а) Трансляторы: доли h_m"),
              (1, ("T1", "T2"), "h_r", "б) Трансляторы: доли h_R"),
              (2, ("AX", "AY"), "h_m", "в) Арбитражёры: доли h_m")]
    for idx, kinds, key, title in panels:
        values = np.array(sorted({a[key] for a in run.agents
                                  if a["kind"] in kinds and a[key] is not None}))
        counts = np.zeros((len(values), run.T + 1))
        alive = run.alive_mask()
        for v, value in enumerate(values):
            rows = [i for i, a in enumerate(run.agents)
                    if a["kind"] in kinds and a[key] == value]
            counts[v] = alive[rows].sum(axis=0)
        total = counts.sum(axis=0)
        shares = np.divide(counts, total, out=np.zeros_like(counts),
                           where=total > 0) * 100
        colors = plt.cm.viridis(np.linspace(0.05, 0.9, len(values)))
        ax[idx].stackplot(t, shares, colors=colors,
                          labels=[f"{key} = {v:g}" for v in values], alpha=0.9)
        ax[idx].axvline(run.cfg.evolution_start, color="black", lw=0.8, ls="--")
        ax[idx].set_ylim(0, 100)
        ax[idx].set_title(title)
        ax[idx].set_xlabel("тик")
        ax[idx].set_ylabel("% живых копий")
        ax[idx].legend(loc="lower left", fontsize=6, ncol=2)
        _thousands(ax[idx])

    alive = run.alive_mask()
    for kind in D.KINDS:
        rows = run.kind_rows(kind)
        ax[3].plot(alive[rows].sum(axis=0), color=KIND_COLOR[kind],
                   label=f"{kind}: {int(alive[rows, run.T].sum())} из {len(rows)}")
    ax[3].axvline(run.cfg.evolution_start, color="black", lw=0.8, ls="--")
    _phase_spans(ax[3], ph)
    ax[3].set_title("г) Число живых копий по типам")
    ax[3].set_xlabel("тик")
    ax[3].set_ylabel("копий")
    ax[3].legend(loc="lower left", fontsize=6.5)
    _thousands(ax[3])
    return fig


def fig_agents_hyper_pnl(run: D.RunData, ph: D.Phases):
    """Связь гиперпараметров с результатом и выживаемостью."""
    cfg = run.cfg
    fig, ax = _fig(2, 2, 4.3)

    for k, kind in enumerate(("T1", "T2")):
        rows = run.kind_rows(kind)
        grid = np.full((len(cfg.h_r_values), len(cfg.h_m_values)), np.nan)
        for i in rows:
            r = list(cfg.h_r_values).index(run.agents[i]["h_r"])
            m = list(cfg.h_m_values).index(run.agents[i]["h_m"])
            grid[r, m] = run.equity[i, run.T]
        # шкала по типичному значению, а не по максимуму: иначе единственный
        # выживший агент насыщает палитру и структура проигравших не видна
        scale = float(np.nanpercentile(np.abs(grid), 75) * 2) or 1.0
        mesh = ax[k].imshow(grid, cmap="RdYlGn", vmin=-scale, vmax=scale,
                            origin="lower", aspect="auto")
        for r in range(grid.shape[0]):
            for m in range(grid.shape[1]):
                dead = any(run.agents[i]["death_tick"] is not None for i in rows
                           if run.agents[i]["h_r"] == cfg.h_r_values[r]
                           and run.agents[i]["h_m"] == cfg.h_m_values[m])
                ax[k].text(m, r, f"{grid[r, m]:+.3f}" + ("\n†" if dead else ""),
                           ha="center", va="center", fontsize=5.5)
        ax[k].set_xticks(range(len(cfg.h_m_values)))
        ax[k].set_xticklabels([f"{v:g}" for v in cfg.h_m_values])
        ax[k].set_yticks(range(len(cfg.h_r_values)))
        ax[k].set_yticklabels([f"{v:g}" for v in cfg.h_r_values])
        ax[k].set_xlabel("h_m")
        ax[k].set_ylabel("h_R")
        ax[k].grid(False)
        ax[k].set_title(f"{'аб'[k]}) {kind}: итоговый PnL по сетке († — выбыл)")
        fig.colorbar(mesh, ax=ax[k], pad=0.01, extend="both").set_label(
            "PnL, X1 (шкала обрезана)", fontsize=6.5)

    width = 0.35
    x = np.arange(len(cfg.h_m_values))
    for j, kind in enumerate(("AX", "AY")):
        rows = run.kind_rows(kind)
        means = [np.mean([run.equity[i, run.T] for i in rows
                          if run.agents[i]["h_m"] == v]) for v in cfg.h_m_values]
        ax[2].bar(x + (j - 0.5) * width, means, width=width,
                  color=KIND_COLOR[kind], label=kind)
    ax[2].axhline(0, color="black", lw=0.7)
    ax[2].set_xticks(x)
    ax[2].set_xticklabels([f"{v:g}" for v in cfg.h_m_values])
    ax[2].set_title("в) Арбитражёры: средний PnL копии по h_m")
    ax[2].set_xlabel("h_m")
    ax[2].set_ylabel("PnL, X1")
    ax[2].legend(loc="upper right", fontsize=6.5)

    # чем отбор руководствовался: результат каждого значения гиперпараметра
    # к моменту включения отбора (по нему и считается убыток за окно)
    warm = D.phase_pnl(run, ph.warmup)
    labels, values_m, values_r = [], [], []
    for value in cfg.h_m_values:
        labels.append(f"{value:g}")
        rows_m = [i for i, a in enumerate(run.agents)
                  if a["h_m"] == value and a["kind"] in ("T1", "T2")]
        rows_r = [i for i, a in enumerate(run.agents)
                  if a["h_r"] == value and a["kind"] in ("T1", "T2")]
        values_m.append(np.mean(warm[rows_m]))
        values_r.append(np.mean(warm[rows_r]) if rows_r else np.nan)
    x = np.arange(len(labels))
    ax[3].bar(x - 0.19, values_m, width=0.36, color="#4c72b0",
              label="по h_m (агрессивность)")
    ax[3].bar(x + 0.19, values_r, width=0.36, color="#dd8452",
              label="по h_R (чувствительность к риску)")
    ax[3].axhline(0, color="black", lw=0.7)
    ax[3].set_xticks(x)
    ax[3].set_xticklabels(labels)
    ax[3].set_title("г) Трансляторы: PnL за прогрев по значению параметра")
    ax[3].set_xlabel("значение гиперпараметра")
    ax[3].set_ylabel("средний PnL копии, X1")
    ax[3].legend(loc="lower left", fontsize=6.5)
    return fig


# --------------------------------------------------------------------------- #
# Реестр фигур и сборка
# --------------------------------------------------------------------------- #

FIGURES = [
    # дидактические
    (ex_auction, FigureSpec(
        "ex_auction", "Батч-аукцион на лимитной бирже",
        "Слева — заявки одного тика, сложенные в кривые спроса и предложения; "
        "цена аукциона p* максимизирует исполняемый объём min(спрос, предложение), "
        "и все сделки тика проходят по ней одной. Справа — та же механика при "
        "сдвинутом якоре генерации: цена аукциона следует за якорем практически "
        "один в один, а объём почти не зависит от сдвига.")),
    (ex_book_formation, FigureSpec(
        "ex_book_formation", "Откуда берутся стакан, спред и глубина",
        "Сторона заявки и её цена разыгрываются независимо, поэтому часть "
        "покупок оказывается выше части продаж — они и порождают сделки, а "
        "неисполненный остаток ложится в стакан (панель б — усреднённый по "
        "прогону профиль). Панель в: чем гуще поток, тем уже спред и больше "
        "глубина; вертикальные штрихи — интенсивности двух площадок прогона.")),
    (ex_coupling, FigureSpec(
        "ex_coupling", "Связь площадок через общий якорь",
        "Разовый шок якоря +2% переносится в цены обеих площадок за несколько "
        "десятков тиков; тонкая биржа шумнее и добирается до нового уровня "
        "неровно. Справа — вес толстой биржи в якоре: он равен её доле глубины, "
        "поэтому тонкая площадка не может утащить общий уровень за собой.")),
    (ex_ce_clearing, FigureSpec(
        "ex_ce_clearing", "Клиринг непрерывной биржи",
        "Две встречные заявки с ценами безразличия 105 и 95: при равных λ "
        "клиринг даёт их геометрическое среднее, при перекосе λ — сдвигается "
        "к более агрессивной стороне (левая панель, логарифмическая ось). "
        "Справа видно, что цена клиринга — это ровно тот уровень, на котором "
        "суммарный поток стоимости обеих заявок обращается в ноль.")),
    (ex_shading, FigureSpec(
        "ex_shading", "Шейдинг котировки против собственной позиции",
        "Цена безразличия z = мид·exp(−g·q) отклоняется от мида тем сильнее, "
        "чем больше накопленная позиция и чем выше h_R (левая панель). Правая "
        "панель переводит это в деньги: при цене CE, равной миду, заявка "
        "создаёт поток λ·f, направленный на возврат позиции к нулю; наклон "
        "задаётся агрессивностью h_m.")),
    (ex_hedge, FigureSpec(
        "ex_hedge", "Порог хеджа: риск против полуспреда",
        "Держать позицию невыгодно, когда риск последней единицы g·|q| "
        "превышает потерю на её сбросе cost = ½·спред/мид; точка равенства "
        "и есть порог |q*| = cost/g (левая панель). Справа — игрушечная "
        "траектория: без хеджа позиция блуждает, с хеджем удерживается "
        "в коридоре, тем более узком, чем выше h_R.")),
    # мир
    (fig_world_price, FigureSpec(
        "world_price", "Справедливая цена и цены обычных бирж",
        "Справедливая цена — экзогенный след новостей, общий для обеих "
        "площадок; якорь — то, как рынок её отслеживает. Миды обеих бирж идут "
        "за якорем, отклоняясь на величину порядка своей микроструктуры: "
        "толстая биржа держится плотно, тонкая гуляет заметно шире (панели в, г).")),
    (fig_world_book, FigureSpec(
        "world_book", "Поведение стаканов обычных бирж",
        "Тепловые карты (а, б) показывают, где в каждый момент лежит объём "
        "относительно мида: у толстой биржи книга плотная и симметричная, у "
        "тонкой — рваная, с провалами и односторонними состояниями. Панель в — "
        "усреднённый профиль, панель г — спред во времени: именно он входит "
        "в качество книги H и в стоимость хеджа.")),
    (fig_world_volume, FigureSpec(
        "world_volume", "Объёмы торгов на обычных биржах",
        "Объём аукциона за тик и накопленный оборот (а, б) прямо пропорциональны "
        "интенсивности фонового потока. Панель г отвечает на важный для модели "
        "вопрос: хеджи трансляторов — это лишь малая доля оборота площадки, "
        "поэтому «бумажный» хедж без влияния на рынок остаётся допустимым "
        "приближением.")),
    (fig_world_liquidity, FigureSpec(
        "world_liquidity", "Ликвидность и глубина рынка",
        "Глубина у мида (а) и качество книги H (б) — это ровно те величины, "
        "из которых складывается торговая мощность семейства трансляторов "
        "Λ = κ_T·H. Панель в — цена немедленного исполнения в зависимости от "
        "размера: на тонкой бирже она растёт быстрее и глубины хватает не "
        "всегда. Панель г сравнивает рыночную волатильность с волатильностью "
        "справедливой цены: микроструктурный шум на порядок больше новостного.")),
    # непрерывная биржа
    (fig_ce_price, FigureSpec(
        "ce_price", "Цена непрерывной биржи против мира",
        "Панели а и б — один и тот же срез в двух режимах: в прогреве "
        "гиперпараметры ещё не отобраны и цена CE следует за мидом менее "
        "уверенно, в стационаре трансляция заметно точнее. Панель в даёт "
        "ошибку трансляции во времени, панель г — её распределение по фазам: "
        "видно и смещение, и разброс.")),
    (fig_ce_basis, FigureSpec(
        "ce_basis", "Базисы арбитражёров на непрерывной бирже",
        "Арбитражёры удерживают курс кассовых активов двух площадок около "
        "паритета (а) и стягивают базисы X1/X2 и Y1/Y2 к нулю (б, в). "
        "Панель г показывает, какое расхождение реально видит заявка каждого "
        "типа: у арбитражёров оно на порядки меньше, чем у трансляторов, — "
        "они работают с почти закрытым базисом.")),
    (fig_ce_clearing, FigureSpec(
        "ce_clearing", "Оборот и качество клиринга непрерывной биржи",
        "Оборот CE и его разбивка по типам (а, б) показывают, кто на самом деле "
        "двигает цены. Панель в — наполнение книги: ранг говорит, сколько "
        "ценовых направлений книга определяет в каждый тик. Панель г — "
        "контрольная: и невязка клиринга, и невязка разложения PnL держатся "
        "на уровне машинной точности, то есть стоимость в системе не "
        "теряется и не создаётся.")),
    # агенты
    (fig_agents_pnl, FigureSpec(
        "agents_pnl", "PnL агентов и их копий",
        "Панель а — все 64 траектории: до включения отбора живут все копии, "
        "дальше проигрывающие замораживаются. Панель в — итог по копиям с "
        "отметкой выбывших, панель г сравнивает заработок в прогреве с "
        "заработком в стационаре: точки в правом верхнем углу — те, кто был "
        "прибыльным в обоих режимах.")),
    (fig_agents_turnover, FigureSpec(
        "agents_turnover", "Объёмы торгов в разрезе агентов и копий",
        "Панель а — не ошибка рисунка: четыре линии совпадают, потому что "
        "условие нулевого потока стоимости связывает нотионалы семейств "
        "в кольцо (см. раздел 4), и каждое семейство даёт ровно четверть "
        "оборота CE — это же видно на панели г. Панель б показывает, как "
        "оборот распределён между копиями (штриховка — выбывшие), панель в — "
        "что копии ложатся на закон λ ∝ 1/h_m.")),
    (fig_agents_structure, FigureSpec(
        "agents_structure", "Структурная аналитика PnL агентов",
        "Разложение точное: PnL = переоценка позиции + результат хеджа "
        "(сделки на CE самофинансируемы и в момент исполнения стоимости не "
        "создают). Панель а показывает обе компоненты накопленным итогом, "
        "панель б — итог по типам, панель в — каждого транслятора в "
        "координатах «переоценка / хедж», панель г связывает результат с "
        "размером удерживаемой позиции.")),
    (fig_agents_inventory, FigureSpec(
        "agents_inventory", "Инвентарь агентов и работа хеджа",
        "Порог хеджа (штриховая линия на панели а — медиана по блокам, "
        "бледная линия — сам ряд) обратно пропорционален дисперсии, поэтому "
        "он не постоянен: на всплесках волатильности он проваливается на "
        "два порядка и дотягивается до текущей позиции — именно в эти "
        "моменты хедж и срабатывает. В остальное время позицию у нуля "
        "держит один лишь шейдинг котировки. Инвентарь записан на конец "
        "тика, то есть уже после хеджа. Панель в — распределение |позиции| "
        "в стационаре, панель г — частота хеджей и их итог по площадкам.")),
    (fig_agents_hyper, FigureSpec(
        "agents_hyper", "Доли гиперпараметров в живых копиях",
        "Состав популяции во времени: до тика включения отбора доли равны "
        "по построению, дальше отбор перераспределяет их. Панели а–в "
        "показывают, какие значения h_m и h_R выживают, панель г — сколько "
        "копий каждого типа осталось.")),
    (fig_agents_hyper_pnl, FigureSpec(
        "agents_hyper_pnl", "Гиперпараметры, результат и выживаемость",
        "Тепловые карты (а, б) дают итоговый PnL транслятора в каждой точке "
        "сетки (h_m, h_R); цветовая шкала обрезана по типичному значению, "
        "иначе единственный выживший насыщает палитру и структура "
        "проигравших не видна — числа в клетках точные. Панель в — средний "
        "PnL арбитражёра в разрезе агрессивности, панель г — результат "
        "трансляторов к моменту включения отбора: именно на эти числа "
        "отбор и реагирует.")),
]


def build_all(run: D.RunData, outdir: Path, verbose: bool = True) -> list:
    """Строит все фигуры в outdir и возвращает их спецификации."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ph = D.phases(run)
    specs = []
    for builder, spec in FIGURES:
        fig = builder(run, ph)
        path = outdir / f"{spec.key}.pdf"
        fig.savefig(path)
        plt.close(fig)
        if verbose:
            print(f"  фигура {spec.key:<20} -> {path.name}")
        specs.append(spec)
    return specs


if __name__ == "__main__":
    run_data = D.load_or_build()
    build_all(run_data, Path(run_data.cfg.report_dir) / "figures")
