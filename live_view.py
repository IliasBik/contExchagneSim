"""
Живая визуализация пары связанных бирж (coupled_market.py).

Запуск:  python live_view.py

Каждый кадр делает несколько тиков рынка и перерисовывает шесть панелей:
    1) цены обеих бирж и якорь;
    2) разница цен A - B (видно притяжение к нулю);
    3) волатильность обеих бирж;
    4) стоимость немедленного хеджа в зависимости от его размера;
    5) стакан биржи A (горизонтальная гистограмма объёмов по ценам);
    6) стакан биржи B.

Окно закрыть — программа завершится.
"""

import matplotlib.pyplot as plt
import numpy as np

from coupled_market import CoupledMarket, ExchangeConfig

# ---------------------------------------------------------------------------
# Параметры симуляции и визуализации — правятся здесь
# ---------------------------------------------------------------------------

CONFIG_A = ExchangeConfig(name="A", arrival_rate=40.0, order_size=1.0,
                          order_ttl=20, price_std=0.5, ewma_half_life=20.0)
CONFIG_B = ExchangeConfig(name="B", arrival_rate=8.0, order_size=1.0,
                          order_ttl=20, price_std=0.5, ewma_half_life=20.0)

TICK_SIZE = 0.01
INITIAL_PRICE = 100.0
ANCHOR_EWMA_HALF_LIFE = 20.0
DEPTH_BAND = 0.5

WARMUP_TICKS = 300        # прогрев пустых стаканов до начала отрисовки
TICKS_PER_FRAME = 5       # тиков рынка на один кадр
HISTORY_WINDOW = 500      # сколько последних тиков держать на графиках
HEDGE_SIZES = np.arange(1, 61)   # сетка размеров хеджа для панели стоимости
BOOK_BIN_SIZE = 0.05      # ширина ценового бина в гистограмме стакана

COLOR_A = "tab:blue"
COLOR_B = "tab:orange"


# ---------------------------------------------------------------------------
# Отрисовка отдельных панелей
# ---------------------------------------------------------------------------

def draw_prices(ax, ticks, prices_a, prices_b, anchors):
    """Панель 1: последние цены аукционов обеих бирж и якорь."""
    ax.clear()
    ax.plot(ticks, prices_a, color=COLOR_A, label="A (толстая)")
    ax.plot(ticks, prices_b, color=COLOR_B, label="B (тонкая)")
    ax.plot(ticks, anchors, color="gray", linewidth=1,
            linestyle="--", label="якорь")
    ax.set_title("Цена последнего аукциона")
    ax.set_xlabel("тик")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)


def draw_price_gap(ax, ticks, prices_a, prices_b):
    """Панель 2: разница цен A - B — видно, что гэп притягивается к нулю."""
    ax.clear()
    gap = np.array(prices_a) - np.array(prices_b)
    ax.plot(ticks, gap, color="tab:green")
    ax.axhline(0.0, color="gray", linewidth=1)
    ax.set_title(f"Разница цен A - B (текущая: {gap[-1]:+.3f})")
    ax.set_xlabel("тик")
    ax.grid(alpha=0.3)


def draw_volatility(ax, ticks, vols_a, vols_b):
    """Панель 3: EWMA-волатильность мида за тик на обеих биржах."""
    ax.clear()
    ax.plot(ticks, vols_a, color=COLOR_A, label=f"A: {vols_a[-1]:.5f}")
    ax.plot(ticks, vols_b, color=COLOR_B, label=f"B: {vols_b[-1]:.5f}")
    ax.set_title("Волатильность (за тик)")
    ax.set_xlabel("тик")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)


def draw_hedge_cost(ax, market):
    """Панель 4: стоимость немедленного хеджа в зависимости от размера.

    Стоимость меряется как проскальзывание средней цены исполнения
    относительно мида биржи (для покупки avg - mid, для продажи mid - avg) —
    в неё входит полспреда и проход по глубине стакана. Линия обрывается
    там, где видимой глубины уже не хватает на полный объём.
    """
    ax.clear()
    for name, color in (("A", COLOR_A), ("B", COLOR_B)):
        exchange = market.exchanges[name]
        mid = exchange.mid
        for side, style, sign in (("buy", "-", +1.0), ("sell", "--", -1.0)):
            sizes, slippages = [], []
            for size in HEDGE_SIZES:
                quote = exchange.quote(side, float(size))
                if quote["filled"] < size - 1e-9:
                    break                      # глубина кончилась — обрыв
                sizes.append(size)
                slippages.append(sign * (quote["avg_price"] - mid))
            ax.plot(sizes, slippages, style, color=color,
                    label=f"{name} {side}")
    ax.set_title("Стоимость хеджа: проскальзывание от мида")
    ax.set_xlabel("размер хеджа")
    ax.set_ylabel("проскальзывание")
    ax.legend(loc="upper left", fontsize=8, ncols=2)
    ax.grid(alpha=0.3)


def draw_book(ax, market, name, color):
    """Панели 5-6: стакан — горизонтальная гистограмма объёмов по ценам.

    Уровни группируются в ценовые бины ширины BOOK_BIN_SIZE, иначе при
    мелком tick_size столбики сливаются в линии. Покупки зелёным, продажи
    красным; горизонтальные линии — мид и последняя цена; в заголовке спред.
    """
    ax.clear()
    exchange = market.exchanges[name]
    depth = exchange.depth_profile()

    for side, bar_color in (("buy", "tab:green"), ("sell", "tab:red")):
        bins: dict[float, float] = {}
        for price, volume in depth[side]:
            bin_center = round(price / BOOK_BIN_SIZE) * BOOK_BIN_SIZE
            bins[bin_center] = bins.get(bin_center, 0.0) + volume
        if bins:
            ax.barh(list(bins.keys()), list(bins.values()),
                    height=BOOK_BIN_SIZE * 0.9, color=bar_color,
                    alpha=0.7, linewidth=0)

    ax.axhline(exchange.mid, color="gray", linewidth=1, linestyle="--")
    ax.axhline(exchange.last_price, color=color, linewidth=1)
    spread = exchange.spread
    spread_text = f"{spread:.2f}" if spread is not None else "-"
    ax.set_title(f"Стакан {name} (спред {spread_text})")
    ax.set_xlabel("объём")
    ax.set_ylabel("цена")
    ax.grid(alpha=0.3)


# ---------------------------------------------------------------------------
# Основной цикл: тики рынка -> обновление истории -> перерисовка
# ---------------------------------------------------------------------------

def main():
    market = CoupledMarket(CONFIG_A, CONFIG_B, tick_size=TICK_SIZE,
                           initial_price=INITIAL_PRICE,
                           anchor_ewma_half_life=ANCHOR_EWMA_HALF_LIFE,
                           depth_band=DEPTH_BAND)
    market.warmup(WARMUP_TICKS)

    # история для линейных графиков (обрезается до HISTORY_WINDOW)
    history = {"tick": [], "price_a": [], "price_b": [], "anchor": [],
               "vol_a": [], "vol_b": []}

    plt.ion()
    figure = plt.figure(figsize=(15, 8))
    grid = figure.add_gridspec(2, 3)
    ax_price = figure.add_subplot(grid[0, 0])
    ax_gap = figure.add_subplot(grid[1, 0])
    ax_vol = figure.add_subplot(grid[1, 1])
    ax_hedge = figure.add_subplot(grid[0, 1])
    ax_book_a = figure.add_subplot(grid[0, 2])
    ax_book_b = figure.add_subplot(grid[1, 2])
    figure.suptitle("Пара связанных бирж: A — толстая, B — тонкая")

    while plt.fignum_exists(figure.number):
        # несколько тиков рынка на кадр
        for _ in range(TICKS_PER_FRAME):
            market.step()
            state = market.get_state()
            history["tick"].append(state["tick"])
            history["anchor"].append(state["anchor"])
            history["price_a"].append(state["exchanges"]["A"]["last_price"])
            history["price_b"].append(state["exchanges"]["B"]["last_price"])
            history["vol_a"].append(state["exchanges"]["A"]["volatility"])
            history["vol_b"].append(state["exchanges"]["B"]["volatility"])

        # держим на графиках только последние HISTORY_WINDOW тиков
        for key in history:
            history[key] = history[key][-HISTORY_WINDOW:]

        draw_prices(ax_price, history["tick"], history["price_a"],
                    history["price_b"], history["anchor"])
        draw_price_gap(ax_gap, history["tick"], history["price_a"],
                       history["price_b"])
        draw_volatility(ax_vol, history["tick"], history["vol_a"],
                        history["vol_b"])
        draw_hedge_cost(ax_hedge, market)
        draw_book(ax_book_a, market, "A", COLOR_A)
        draw_book(ax_book_b, market, "B", COLOR_B)

        figure.tight_layout(rect=(0, 0, 1, 0.96))
        plt.pause(0.5)   # отдаём управление окну и держим темп анимации


if __name__ == "__main__":
    main()
