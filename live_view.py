"""
Живая визуализация пары связанных бирж (coupled_market.py).

Запуск:  python live_view.py

Два режима (переключатель IMPACT_DEMO ниже):

  IMPACT_DEMO = True  — демонстрация прайс-импакта: глубокая книга с
      памятью (order_ttl=20, плотный поток, узкий разброс цен, медленный
      якорь, фундаментальный дрейф выключен). Ударьте по тонкой бирже 2
      объёмом ~30-50 — мид подпрыгнет и будет плавно возвращаться ~10-15
      тиков, пока фоновый поток заново заполняет выеденную сторону стакана.
      Толстая биржа 1 тот же удар почти поглощает — тоже поучительно.

  IMPACT_DEMO = False — рынок ровно из SimConfig (agent_simulation.py):
      то, на чём торгуют агенты, с теми же параметрами, seed и дрейфом.
      Внимание: при коротком order_ttl книга пересобирается каждый тик,
      у неё нет памяти, и плавного возврата импакта не бывает — удар
      съедает стакан, а следующий аукцион строит рынок заново вокруг
      якоря; шум мида при этом сравним с самим импактом.

Панели: 1) миды бирж, якорь, фундаментал и метки ручных заявок;
        2) разница мидов; 3) волатильность; 4) стоимость немедленного
        хеджа от размера; 5-6) стаканы (в заголовке спред и H).

Внизу — панель ручных заявок: объём, биржа, сторона -> "Отправить".
Заявка проходит через ТОТ ЖЕ механизм, что и реальные хеджи агентов
(CoupledMarket.execute_hedges): рыночный проход по обезличенным заявкам,
исполненное исчезает из стакана, last_price обновляется ценой сделки.
Момент заявки отмечается вертикальной чертой на панели цен (зелёная —
покупка, красная — продажа); в статусе — исполненный объём, средняя цена
и проскальзывание. Кнопка "Пауза" замораживает фоновый поток: можно бить
по стакану и разглядывать вмятину без шума, затем "Пуск".

Окно закрыть — программа завершится.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, RadioButtons, TextBox

import agent_formulas as F
from agent_simulation import SimConfig
from coupled_market import CoupledMarket, ExchangeConfig

# ---------------------------------------------------------------------------
# Режим и параметры рынка
# ---------------------------------------------------------------------------

CFG = SimConfig()
IMPACT_DEMO = True    # False — смотреть ровно рынок симуляции (SimConfig)

if IMPACT_DEMO:
    # книга с памятью: заявки живут долго, поток плотный, разброс узкий,
    # якорь медленный, новостного дрейфа нет — импакт виден в чистом виде
    VENUE1 = ExchangeConfig("1", arrival_rate=40.0, order_size=1.0,
                            order_ttl=40, price_std=1.0, ewma_half_life=20.0)
    VENUE2 = ExchangeConfig("2", arrival_rate=10.0, order_size=1.0,
                            order_ttl=40, price_std=1.0, ewma_half_life=20.0)
    ANCHOR_HALF_LIFE = 50.0
    DEPTH_BAND = 2.0
    FUNDAMENTAL_VOL = 0.0
else:
    VENUE1, VENUE2 = CFG.venue1, CFG.venue2
    ANCHOR_HALF_LIFE = CFG.anchor_half_life
    DEPTH_BAND = CFG.depth_band
    FUNDAMENTAL_VOL = CFG.fundamental_vol

# --- настройки самой картинки ----------------------------------------------

TICKS_PER_FRAME = 1       # тиков рынка на кадр (1 — видно динамику импакта)
FRAME_PAUSE = 0.25        # пауза между кадрами, сек
HISTORY_WINDOW = 300      # сколько последних тиков держать на графиках
HEDGE_SIZES = np.arange(1, 81)   # сетка размеров хеджа для панели стоимости
BOOK_BIN_SIZE = 0.10      # ширина ценового бина в гистограмме стакана

VENUES = ("1", "2")
VENUE_TITLES = {"1": "1 (толстая)", "2": "2 (тонкая)"}
COLORS = {"1": "tab:blue", "2": "tab:orange"}


# ---------------------------------------------------------------------------
# Отрисовка отдельных панелей
# ---------------------------------------------------------------------------

def draw_prices(ax, history, events):
    """Панель 1: миды бирж, якорь, фундаментал и метки ручных заявок.

    Рисуется именно мид: импакт живёт в состоянии стакана. Цена аукциона
    по построению пересобирается за один тик и кривую возврата не покажет.
    """
    ax.clear()
    ticks = history["tick"]
    for v in VENUES:
        ax.plot(ticks, history[f"mid_{v}"], color=COLORS[v],
                label=VENUE_TITLES[v])
    ax.plot(ticks, history["anchor"], color="gray", linewidth=1,
            linestyle="--", label="якорь")
    if FUNDAMENTAL_VOL > 0.0:
        ax.plot(ticks, history["fundamental"], color="black", linewidth=1,
                linestyle=":", label="фундаментал")
    for ev in events:
        color = "tab:green" if ev["side"] == "buy" else "tab:red"
        ax.axvline(ev["tick"], color=color, linewidth=1, alpha=0.6)
    ax.set_title("Мид")
    ax.set_xlabel("тик")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)


def draw_price_gap(ax, history):
    """Панель 2: разница мидов бирж — видно притяжение к нулю."""
    ax.clear()
    gap = np.array(history["mid_1"]) - np.array(history["mid_2"])
    ax.plot(history["tick"], gap, color="tab:green")
    ax.axhline(0.0, color="gray", linewidth=1)
    ax.set_title(f"Разница мидов 1 - 2 (текущая: {gap[-1]:+.3f})")
    ax.set_xlabel("тик")
    ax.grid(alpha=0.3)


def draw_volatility(ax, history):
    """Панель 3: EWMA-волатильность мида за тик на обеих биржах."""
    ax.clear()
    for v in VENUES:
        vols = history[f"vol_{v}"]
        ax.plot(history["tick"], vols, color=COLORS[v],
                label=f"{VENUE_TITLES[v]}: {vols[-1]:.5f}")
    ax.set_title("Волатильность (за тик)")
    ax.set_xlabel("тик")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)


def draw_hedge_cost(ax, market):
    """Панель 4: стоимость немедленного хеджа в зависимости от размера.

    Стоимость меряется как проскальзывание средней цены исполнения
    относительно мида биржи (для покупки avg - mid, для продажи mid - avg) —
    в неё входит полспреда и проход по глубине стакана. Линия обрывается
    там, где видимой глубины уже не хватает на полный объём. Оценка идёт
    через quote — стакан она не трогает.
    """
    ax.clear()
    for v in VENUES:
        exchange = market.exchanges[v]
        mid = exchange.mid
        for side, style, sign in (("buy", "-", +1.0), ("sell", "--", -1.0)):
            sizes, slippages = [], []
            for size in HEDGE_SIZES:
                quote = exchange.quote(side, float(size))
                if quote["filled"] < size - 1e-9:
                    break                      # глубина кончилась — обрыв
                sizes.append(size)
                slippages.append(sign * (quote["avg_price"] - mid))
            ax.plot(sizes, slippages, style, color=COLORS[v],
                    label=f"{v} {side}")
    ax.set_title("Стоимость хеджа: проскальзывание от мида")
    ax.set_xlabel("размер хеджа")
    ax.set_ylabel("проскальзывание")
    ax.legend(loc="upper left", fontsize=8, ncols=2)
    ax.grid(alpha=0.3)


def draw_book(ax, market, venue):
    """Панели 5-6: стакан — горизонтальная гистограмма объёмов по ценам.

    Уровни группируются в ценовые бины ширины BOOK_BIN_SIZE, иначе при
    мелком tick_size столбики сливаются в линии. Покупки зелёным, продажи
    красным; горизонтальные линии — мид и последняя цена. В заголовке спред
    и H — качество книги глазами трансляторов (см. agent_formulas).
    """
    ax.clear()
    exchange = market.exchanges[venue]
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
    ax.axhline(exchange.last_price, color=COLORS[venue], linewidth=1)
    spread = exchange.spread
    spread_text = f"{spread:.2f}" if spread is not None else "-"
    H = F.book_quality(exchange.depth_near_mid(DEPTH_BAND), exchange.mid,
                       spread, CFG.tick_size)
    ax.set_title(f"Стакан {venue} (спред {spread_text}, H={H:.1f})")
    ax.set_xlabel("объём")
    ax.set_ylabel("цена")
    ax.grid(alpha=0.3)


# ---------------------------------------------------------------------------
# Основной цикл: тики рынка -> обновление истории -> перерисовка
# ---------------------------------------------------------------------------

def build_market() -> CoupledMarket:
    """Рынок по выбранному режиму (импакт-демо или ровно SimConfig)."""
    market = CoupledMarket(
        VENUE1, VENUE2, tick_size=CFG.tick_size,
        initial_price=CFG.initial_price,
        anchor_ewma_half_life=ANCHOR_HALF_LIFE,
        depth_band=DEPTH_BAND, seed=CFG.seed,
        fundamental_vol=FUNDAMENTAL_VOL)
    market.warmup(max(CFG.warmup, 3 * VENUE1.order_ttl))
    return market


def main(max_frames: int | None = None):
    market = build_market()

    # история для линейных графиков (обрезается до HISTORY_WINDOW)
    history = {"tick": [], "anchor": [], "fundamental": []}
    for v in VENUES:
        history[f"mid_{v}"] = []
        history[f"vol_{v}"] = []
    events: list[dict] = []          # ручные заявки для меток на графике
    ui = {"paused": False}

    plt.ion()
    figure = plt.figure(figsize=(15, 9))
    grid = figure.add_gridspec(2, 3)
    ax_price = figure.add_subplot(grid[0, 0])
    ax_gap = figure.add_subplot(grid[1, 0])
    ax_vol = figure.add_subplot(grid[1, 1])
    ax_hedge = figure.add_subplot(grid[0, 1])
    ax_books = {"1": figure.add_subplot(grid[0, 2]),
                "2": figure.add_subplot(grid[1, 2])}
    mode = "импакт-демо" if IMPACT_DEMO else "рынок симуляции (SimConfig)"
    figure.suptitle(f"Пара связанных бирж: 1 — толстая, 2 — тонкая  [{mode}]")
    figure.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.24,
                           hspace=0.45, wspace=0.30)

    # --- панель ручных заявок ---------------------------------------------- #
    box_size = TextBox(figure.add_axes([0.10, 0.06, 0.06, 0.05]),
                       "объём ", initial="40")
    rb_venue = RadioButtons(figure.add_axes([0.19, 0.02, 0.10, 0.11]),
                            ("биржа 1", "биржа 2"), active=1)
    rb_side = RadioButtons(figure.add_axes([0.31, 0.02, 0.10, 0.11]),
                           ("купить", "продать"))
    btn_send = Button(figure.add_axes([0.43, 0.06, 0.12, 0.05]),
                      "Отправить заявку")
    btn_pause = Button(figure.add_axes([0.57, 0.06, 0.08, 0.05]), "Пауза")
    hint = ("Для кривой импакта: биржа 2, объём ~30-50 — мид подпрыгнет "
            "и стечёт обратно за 10-15 тиков." if IMPACT_DEMO else
            "Рыночная заявка бьёт по стакану через execute_hedges — "
            "как хеджи агентов.")
    status = figure.text(0.67, 0.10, hint, fontsize=9, va="top", wrap=True)

    def on_send(_event):
        try:
            size = float(box_size.text.replace(",", "."))
        except ValueError:
            status.set_text("Объём не число.")
            return
        if size <= 0.0:
            status.set_text("Объём должен быть > 0.")
            return
        venue = "1" if rb_venue.value_selected == "биржа 1" else "2"
        side = "buy" if rb_side.value_selected == "купить" else "sell"
        exchange = market.exchanges[venue]
        mid_before = exchange.mid
        fill = market.execute_hedges(venue, [("ручная", side, size)])["ручная"]
        if fill["filled"] <= 0.0:
            status.set_text(f"Биржа {venue}: не исполнилось — встречная "
                            f"сторона стакана пуста.")
            return
        sign = +1.0 if side == "buy" else -1.0
        slippage = sign * (fill["avg_price"] - mid_before)
        events.append({"tick": market.tick, "side": side})
        status.set_text(
            f"Биржа {venue}, {'покупка' if side == 'buy' else 'продажа'} "
            f"{size:g}: исполнено {fill['filled']:g} @ "
            f"{fill['avg_price']:.3f} (мид был {mid_before:.3f}, "
            f"проскальзывание {slippage:+.3f}); mid теперь {exchange.mid:.3f}")

    def on_pause(_event):
        ui["paused"] = not ui["paused"]
        btn_pause.label.set_text("Пуск" if ui["paused"] else "Пауза")

    btn_send.on_clicked(on_send)
    btn_pause.on_clicked(on_pause)

    frame = 0
    while plt.fignum_exists(figure.number):
        if not ui["paused"]:
            # несколько тиков рынка на кадр
            for _ in range(TICKS_PER_FRAME):
                market.step()
                history["tick"].append(market.tick)
                history["anchor"].append(market.anchor)
                history["fundamental"].append(market.fundamental)
                for v in VENUES:
                    ex = market.exchanges[v]
                    history[f"mid_{v}"].append(ex.mid)
                    history[f"vol_{v}"].append(ex.volatility)
            # держим на графиках только последние HISTORY_WINDOW тиков
            for key in history:
                history[key] = history[key][-HISTORY_WINDOW:]
            events[:] = [ev for ev in events
                         if ev["tick"] >= history["tick"][0]]

        # книги и стоимость хеджа перерисовываем всегда: на паузе в них
        # видно эффект ручных заявок
        draw_prices(ax_price, history, events)
        draw_price_gap(ax_gap, history)
        draw_volatility(ax_vol, history)
        draw_hedge_cost(ax_hedge, market)
        for v in VENUES:
            draw_book(ax_books[v], market, v)

        plt.pause(FRAME_PAUSE)   # отдаём управление окну, держим темп
        frame += 1
        if max_frames is not None and frame >= max_frames:
            break


if __name__ == "__main__":
    main()
