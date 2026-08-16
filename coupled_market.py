"""
Симулятор пары связанных классических бирж.

Дискретное время: раз в тик на каждой бирже проходит батч-аукцион,
все сделки тика исполняются по единой цене p*. Связь бирж — через общий
якорь генерации фонового потока заявок (EWMA средневзвешенного мидов).

Структура:
    Order         — одна лимитная заявка в стакане
    ExchangeConfig — параметры одной биржи
    Exchange      — одна биржа: стакан, аукцион, метрики, оценка хеджа
    CoupledMarket — пара бирж + общий якорь; внешний интерфейс

Использование из внешнего файла:
    market = CoupledMarket(config_a, config_b, ...)
    market.warmup(200)                       # прогрев пустых стаканов
    for t in range(T):
        market.step()                        # один тик фонового рынка
        state = market.get_state()           # состояние обеих бирж
        result = market.quote_hedge("A", "sell", 25.0)   # оценка хеджа
        # result = {"filled": ..., "total_cost": ..., "avg_price": ...}
"""

from dataclasses import dataclass

import numpy as np

# допуск для сравнения вещественных объёмов
EPS = 1e-12


@dataclass
class Order:
    """Лимитная заявка (лежащая в стакане или пришедшая на текущий аукцион)."""

    side: str        # "buy" | "sell"
    price: float     # цена на дискретной сетке (кратна tick_size)
    size: float      # оставшийся (неисполненный) объём
    birth_tick: int  # тик появления; заявка живёт order_ttl тиков


@dataclass
class ExchangeConfig:
    """Параметры одной биржи."""

    name: str              # имя биржи (ключ во внешнем интерфейсе)
    arrival_rate: float    # среднее число фоновых заявок за тик (Пуассон)
    order_size: float      # фиксированный размер фоновой заявки
    order_ttl: int         # время жизни заявки, в тиках
    price_std: float       # std смещения цены заявки от якоря
    ewma_half_life: float  # полураспад EWMA волатильности, в тиках


class Exchange:
    """Одна классическая биржа: стакан, батч-аукцион, метрики, оценка хеджа."""

    def __init__(self, config: ExchangeConfig, tick_size: float,
                 initial_price: float, rng: np.random.Generator):
        self.config = config
        self.tick_size = tick_size
        self.rng = rng

        self.bids: list[Order] = []   # лежащие заявки на покупку
        self.asks: list[Order] = []   # лежащие заявки на продажу

        self.last_price = initial_price   # цена последнего аукциона (p*)
        self.last_trade_volume = 0.0      # объём последнего аукциона

        # волатильность: EWMA квадратов лог-приростов мида за тик
        self._ewma_alpha = 1.0 - 0.5 ** (1.0 / config.ewma_half_life)
        self._ewma_var = 0.0
        self._prev_mid = initial_price

    # ------------------------------------------------------------------
    # Рыночные метрики
    # ------------------------------------------------------------------

    @property
    def best_bid(self):
        """Лучшая цена на покупку (None, если покупок в стакане нет)."""
        return max((o.price for o in self.bids), default=None)

    @property
    def best_ask(self):
        """Лучшая цена на продажу (None, если продаж в стакане нет)."""
        return min((o.price for o in self.asks), default=None)

    @property
    def spread(self):
        """Спред best_ask - best_bid (None, если одна из сторон пуста)."""
        if self.best_bid is None or self.best_ask is None:
            return None
        return self.best_ask - self.best_bid

    @property
    def mid(self):
        """Середина спреда; во всех спорных случаях — последняя цена аукциона."""
        if self.best_bid is None or self.best_ask is None:
            return self.last_price
        return 0.5 * (self.best_bid + self.best_ask)

    @property
    def volatility(self):
        """Волатильность мида за тик: корень из EWMA квадратов лог-приростов."""
        return float(np.sqrt(self._ewma_var))

    def depth_profile(self):
        """Распределение лежащего объёма по ценовым уровням.

        Возвращает {"buy":  [(цена, объём), ...] по убыванию цены,
                    "sell": [(цена, объём), ...] по возрастанию цены},
        то есть обе стороны — от лучшего уровня вглубь.
        """
        return {
            "buy": self._aggregate_levels(self.bids, best_first_desc=True),
            "sell": self._aggregate_levels(self.asks, best_first_desc=False),
        }

    def depth_near_mid(self, band: float):
        """Суммарный лежащий объём обеих сторон в полосе ±band от мида.

        Используется как вес биржи при построении общего якоря: тонкая
        биржа получает малый вес и не может утащить якорь за собой.
        """
        m = self.mid
        return sum(o.size for o in self.bids + self.asks
                   if abs(o.price - m) <= band)

    @staticmethod
    def _aggregate_levels(orders, best_first_desc: bool):
        """Суммирует объём заявок по ценовым уровням и сортирует уровни."""
        levels: dict[float, float] = {}
        for o in orders:
            levels[o.price] = levels.get(o.price, 0.0) + o.size
        return sorted(levels.items(), reverse=best_first_desc)

    # ------------------------------------------------------------------
    # Жизненный цикл тика: истечение -> генерация -> аукцион -> статистика
    # ------------------------------------------------------------------

    def expire_orders(self, current_tick: int):
        """Удаляет заявки, прожившие order_ttl тиков.

        Правило действует и на остатки частично исполненных заявок:
        возраст считается от тика появления, а не от последнего исполнения.
        """
        ttl = self.config.order_ttl
        self.bids = [o for o in self.bids if current_tick - o.birth_tick < ttl]
        self.asks = [o for o in self.asks if current_tick - o.birth_tick < ttl]

    def generate_background_orders(self, anchor: float, current_tick: int):
        """Фоновый поток заявок за один тик.

        Число заявок N ~ Пуассон(arrival_rate). Для каждой заявки сначала
        выбирается сторона (монетка 50/50), затем НЕЗАВИСИМО цена
        ~ Normal(anchor, price_std), округлённая к ценовой сетке и
        ограниченная снизу одним тиком. Благодаря независимости стороны и
        цены часть заявок пересекает стакан — они и порождают сделки.
        """
        n = self.rng.poisson(self.config.arrival_rate)
        orders = []
        for _ in range(n):
            side = "buy" if self.rng.random() < 0.5 else "sell"
            raw_price = self.rng.normal(anchor, self.config.price_std)
            price = max(round(raw_price / self.tick_size) * self.tick_size,
                        self.tick_size)
            orders.append(Order(side, price, self.config.order_size,
                                current_tick))
        return orders

    def run_auction(self, new_orders: list[Order]):
        """Батч-аукцион: все сделки тика проходят по единой цене p*.

        p* максимизирует исполняемый объём min(спрос(p), предложение(p)).
        При нескольких оптимумах берётся цена, ближайшая к последней
        (при равном расстоянии — меньшая). Если пересечения нет, сделок
        нет и last_price не меняется; новые заявки просто ложатся в стакан.
        """
        buys = self.bids + [o for o in new_orders if o.side == "buy"]
        sells = self.asks + [o for o in new_orders if o.side == "sell"]

        p_star, volume = self._find_clearing_price(buys, sells)

        if p_star is None:
            self.last_trade_volume = 0.0
        else:
            self._fill_side(buys, "buy", p_star, volume)
            self._fill_side(sells, "sell", p_star, volume)
            self.last_price = p_star
            self.last_trade_volume = volume

        # в стакан возвращается всё неисполненное (включая остатки)
        self.bids = [o for o in buys if o.size > EPS]
        self.asks = [o for o in sells if o.size > EPS]

    def _find_clearing_price(self, buys, sells):
        """Ищет цену аукциона.

        Возвращает (p*, объём) либо (None, 0.0), если сделок нет.
        Кандидаты — ценовые уровни существующих заявок: между уровнями
        объём не меняется, поэтому других цен рассматривать не нужно.
        """
        if not buys or not sells:
            return None, 0.0

        candidate_prices = sorted({o.price for o in buys} |
                                  {o.price for o in sells})
        best_price, best_volume = None, 0.0
        for p in candidate_prices:
            demand = sum(o.size for o in buys if o.price >= p)
            supply = sum(o.size for o in sells if o.price <= p)
            volume = min(demand, supply)
            if volume > best_volume + EPS:
                best_price, best_volume = p, volume
            elif (best_price is not None and abs(volume - best_volume) <= EPS
                  and abs(p - self.last_price)
                  < abs(best_price - self.last_price) - EPS):
                # тай-брейк: тот же объём, но цена ближе к последней
                best_price = p

        if best_volume <= EPS:
            return None, 0.0
        return best_price, best_volume

    @staticmethod
    def _fill_side(orders, side: str, p_star: float, volume: float):
        """Исполняет объём volume на одной стороне аукциона.

        Приоритет по цене: уровни лучше p* исполняются первыми и полностью;
        на уровне, где объёма на всех не хватает (обычно это ровно p*),
        заявки исполняются pro-rata — так ни у кого нет преимущества
        от порядка прихода.
        """
        if side == "buy":
            eligible = [o for o in orders if o.price >= p_star]
            best_first_desc = True    # для покупок лучший уровень — выше
        else:
            eligible = [o for o in orders if o.price <= p_star]
            best_first_desc = False   # для продаж лучший уровень — ниже

        levels: dict[float, list[Order]] = {}
        for o in eligible:
            levels.setdefault(o.price, []).append(o)

        remaining = volume
        for price in sorted(levels, reverse=best_first_desc):
            level_orders = levels[price]
            level_volume = sum(o.size for o in level_orders)
            if level_volume <= remaining + EPS:
                # уровень исполняется целиком
                for o in level_orders:
                    o.size = 0.0
                remaining -= level_volume
            else:
                # объёма на всех не хватает — pro-rata
                fill_share = remaining / level_volume
                for o in level_orders:
                    o.size *= (1.0 - fill_share)
                remaining = 0.0
            if remaining <= EPS:
                break

    def update_stats(self):
        """Обновляет EWMA-волатильность по лог-приросту мида за тик."""
        m = self.mid
        log_return = np.log(m / self._prev_mid)
        self._ewma_var = ((1.0 - self._ewma_alpha) * self._ewma_var
                          + self._ewma_alpha * log_return ** 2)
        self._prev_mid = m

    # ------------------------------------------------------------------
    # Интерфейс для агентов: оценка хеджа (v1) и реальная торговля (v2)
    # ------------------------------------------------------------------

    def quote(self, side: str, size: float):
        """Оценка немедленного хеджа объёма size БЕЗ влияния на рынок.

        side — сторона агента: "buy" идёт по заявкам на продажу,
        "sell" — по заявкам на покупку. Проход от лучшей цены вглубь
        текущего (остаточного после аукциона) стакана. Если видимой
        глубины не хватает, исполняется только часть заявки.

        Возвращает словарь:
            filled     — исполненный объём (<= size; 0, если стакан пуст)
            total_cost — суммарная стоимость исполненной части
            avg_price  — средняя цена исполненной части (None при filled=0)

        Оценка не учитывает одновременные хеджи других агентов и изменение
        стакана к следующему аукциону — это цена по книге "здесь и сейчас".
        """
        book = self.asks if side == "buy" else self.bids
        # уровни от лучшего вглубь: покупаем — от дешёвых продаж вверх,
        # продаём — от дорогих покупок вниз
        levels = self._aggregate_levels(book, best_first_desc=(side == "sell"))

        filled, total_cost = 0.0, 0.0
        for price, level_volume in levels:
            take = min(size - filled, level_volume)
            filled += take
            total_cost += take * price
            if filled >= size - EPS:
                break

        avg_price = total_cost / filled if filled > EPS else None
        return {"filled": filled, "total_cost": total_cost,
                "avg_price": avg_price}

    def execute(self, side: str, size: float):
        """Реальное исполнение заявки агента — парный интерфейс к quote.

        В версии v1 агенты не влияют на рынок (хедж «на бумаге» через
        quote), поэтому метод намеренно не реализован. При переходе к v2
        достаточно реализовать его и заменить вызов quote -> execute.
        """
        raise NotImplementedError(
            "Реальная торговля агентов будет добавлена в следующей версии")


class CoupledMarket:
    """Пара связанных бирж с общим якорем генерации фонового потока.

    Порядок операций внутри тика (step):
        1) истечение заявок на обеих биржах;
        2) пересчёт якоря по состоянию стаканов на конец прошлого тика;
        3) генерация фонового потока — обе биржи видят ОДИН и тот же якорь;
        4) батч-аукцион на каждой бирже независимо;
        5) обновление статистики (волатильность).
    """

    def __init__(self, config_a: ExchangeConfig, config_b: ExchangeConfig,
                 tick_size: float, initial_price: float,
                 anchor_ewma_half_life: float, depth_band: float,
                 seed: int | None = None):
        """
        config_a, config_b    — параметры двух бирж (ExchangeConfig)
        tick_size             — шаг ценовой сетки (общий для обеих бирж)
        initial_price         — стартовая цена: якорь, last_price и мид при t=0
        anchor_ewma_half_life — полураспад EWMA якоря, в тиках
        depth_band            — полоса ±depth_band вокруг мида, в которой
                                считается глубина для весов якоря
        seed                  — зерно генератора случайных чисел (None —
                                невоспроизводимый запуск)
        """
        rng = np.random.default_rng(seed)
        self.exchanges = {
            config_a.name: Exchange(config_a, tick_size, initial_price, rng),
            config_b.name: Exchange(config_b, tick_size, initial_price, rng),
        }
        self.tick = 0
        self.depth_band = depth_band
        self.anchor = initial_price
        self._anchor_alpha = 1.0 - 0.5 ** (1.0 / anchor_ewma_half_life)

    def _update_anchor(self):
        """Якорь = EWMA средневзвешенного мидов; веса — глубина у мида.

        Спорные случаи закрыты правилом «последняя цена»: мид пустой
        стороны сам откатывается к last_price биржи, а при нулевой
        суммарной глубине (пустые стаканы) веса берутся равными.
        """
        mids, weights = [], []
        for ex in self.exchanges.values():
            mids.append(ex.mid)
            weights.append(ex.depth_near_mid(self.depth_band))

        total_weight = sum(weights)
        if total_weight <= EPS:
            weighted_mid = sum(mids) / len(mids)
        else:
            weighted_mid = sum(m * w for m, w in zip(mids, weights)) / total_weight

        self.anchor = ((1.0 - self._anchor_alpha) * self.anchor
                       + self._anchor_alpha * weighted_mid)
        return self.anchor

    def step(self):
        """Один тик фонового рынка (агенты в v1 в тике не участвуют)."""
        self.tick += 1
        for ex in self.exchanges.values():
            ex.expire_orders(self.tick)
        anchor = self._update_anchor()
        for ex in self.exchanges.values():
            new_orders = ex.generate_background_orders(anchor, self.tick)
            ex.run_auction(new_orders)
        for ex in self.exchanges.values():
            ex.update_stats()

    def warmup(self, n_ticks: int):
        """Прогрев: стаканы начинаются пустыми, статистика первых тиков
        нерепрезентативна — прогоняем n_ticks фонового потока вхолостую."""
        for _ in range(n_ticks):
            self.step()

    def get_state(self):
        """Текущее состояние рынка по обеим биржам.

        Возвращает {"tick", "anchor", "exchanges": {имя: {...}}}, где по
        каждой бирже: last_price, last_trade_volume, best_bid, best_ask,
        spread, mid, volatility (за тик), depth (см. depth_profile).
        """
        state = {"tick": self.tick, "anchor": self.anchor, "exchanges": {}}
        for name, ex in self.exchanges.items():
            state["exchanges"][name] = {
                "last_price": ex.last_price,
                "last_trade_volume": ex.last_trade_volume,
                "best_bid": ex.best_bid,
                "best_ask": ex.best_ask,
                "spread": ex.spread,
                "mid": ex.mid,
                "volatility": ex.volatility,
                "depth": ex.depth_profile(),
            }
        return state

    def quote_hedge(self, exchange_name: str, side: str, size: float):
        """Оценка хеджа на бирже exchange_name; подробности — Exchange.quote."""
        return self.exchanges[exchange_name].quote(side, size)


if __name__ == "__main__":
    # мини-демонстрация: толстая биржа A и тонкая B
    config_a = ExchangeConfig(name="A", arrival_rate=40.0, order_size=1.0,
                              order_ttl=20, price_std=0.5, ewma_half_life=20.0)
    config_b = ExchangeConfig(name="B", arrival_rate=8.0, order_size=1.0,
                              order_ttl=20, price_std=0.5, ewma_half_life=20.0)
    market = CoupledMarket(config_a, config_b, tick_size=0.01,
                           initial_price=100.0, anchor_ewma_half_life=20.0,
                           depth_band=0.5)
    market.warmup(200)
    for _ in range(300):
        market.step()

    s = market.get_state()
    for name in ("A", "B"):
        e = s["exchanges"][name]
        print(f"{name}: last={e['last_price']:.2f} "
              f"bid={e['best_bid']} ask={e['best_ask']} "
              f"spread={e['spread']} vol={e['volatility']:.5f}")
    print("якорь:", round(s["anchor"], 3))
    print("хедж sell 25 на A:", market.quote_hedge("A", "sell", 25.0))
    print("хедж sell 25 на B:", market.quote_hedge("B", "sell", 25.0))
