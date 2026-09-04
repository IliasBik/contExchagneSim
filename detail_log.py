"""
detail_log.py — детальный потиковый журнал прогона в JSONL.

Назначение: при малой популяции (единицы агентов) писать ВСЁ — каждую
поданную заявку, каждый филл клиринга, каждый хедж и полное состояние
каждого агента на каждом тике. Формат — одна строка JSON на тик; удобно
обрабатывать построчно или разворачивать в таблицы pandas.

Файл начинается со строки {"type": "meta", ...} — конфигурация прогона,
список агентов, калиброванная gamma. Дальше идут строки {"type": "tick"}:

    t       — номер тика (0 — состояние сразу после прогрева, до торговли)
    world   — фундаментальная цена, якорь и состояние обеих лимитных бирж:
              mid, last, best bid/ask, спред, глубина у мида, EWMA-vol,
              объём последнего аукциона, схема стакана (BOOK_LEVELS лучших
              ценовых уровней на сторону: [[цена, объём], ...]) на конец
              тика; на тиках с хеджами дополнительно before_hedge — mid,
              last, спред, best bid/ask и схема стакана ДО батча хеджей
              (реальный хедж съедает книгу и двигает last_price)
    ce      — цены CE после клиринга и итоги клиринга: оборот, ранг книги,
              невязка, итерации кап-обрезки, чистые потоки стоимости и
              количества по активам
    orders  — заявки, поданные на этом тике: oid, агент, тип, полный спек
              (weights, z, lam, lam_ccy, expiry) и meta — наблюдаемые
              величины на момент подачи (mid, H, capital, g, инвентарь...)
    fills   — филлы клиринга: oid (сшивается с orders), f, нотионал,
              количества по каждой ноге
    hedges  — хеджи об домашний стакан (v2 — батч execute_hedges, филлы
              pro-rata по средней цене стороны): сторона, запрошено/
              исполнено, средняя цена, стоимость исполнения, вклад в PnL

Подключение не требует правки логики симуляции: attach() оборачивает
ce.submit, market.execute_hedges (и quote_hedge для v1) и rec.hedge
и снимает копию событий по пути.
    agents  — состояние каждого агента ПОСЛЕ тика: балансы по всем активам,
              equity (PnL в X1), капитал, инвентарь торгуемой ноги, жив/мёртв

Пример обработки:

    import json
    import pandas as pd

    rows = [json.loads(line)
            for line in open("detail_log.jsonl", encoding="utf-8")]
    meta, ticks = rows[0], rows[1:]
    orders = pd.json_normalize(ticks, "orders", ["t"])
    fills  = pd.json_normalize(ticks, "fills",  ["t"])
    hedges = pd.json_normalize(ticks, "hedges", ["t"])
    agents = pd.json_normalize(ticks, "agents", ["t"])
    world  = pd.json_normalize([{"t": r["t"], **r["world"]} for r in ticks])
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np

import agent_formulas as F

VENUES = ("1", "2")
BOOK_LEVELS = 20     # сколько лучших ценовых уровней стакана писать на сторону


def _book_levels(orders, best_first_desc: bool, limit: int = BOOK_LEVELS):
    """Объём по ценовым уровням, от лучшей цены вглубь: [[цена, объём], ...]."""
    levels: dict[float, float] = {}
    for o in orders:
        levels[o.price] = levels.get(o.price, 0.0) + o.size
    ordered = sorted(levels.items(), reverse=best_first_desc)[:limit]
    return [[p, v] for p, v in ordered]


def _jsonable(obj):
    """numpy-скаляры и массивы -> обычные типы Python."""
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"не сериализуется в JSON: {type(obj)!r}")


class DetailLog:
    """Журнал прогона: одна строка JSON на тик, максимально подробно."""

    def __init__(self, path: str, cfg, agents: list, gamma: float):
        self.path = path
        self._cfg = cfg
        self._fh = open(path, "w", encoding="utf-8")
        self._kind = {a.name: a.kind for a in agents}
        self._orders: list[dict] = []   # заявки, поданные с прошлой записи
        self._hedges: list[dict] = []   # хеджи с прошлой записи
        self._last_quote: dict | None = None   # v1: одиночный quote_hedge
        self._batch: dict = {}          # v2: результаты execute_hedges по именам
        self._pre_books: dict = {}      # книга каждой площадки ДО батча хеджей
        self._write({
            "type": "meta",
            "assets": list(F.ASSETS),
            "venues": list(VENUES),
            "gamma": gamma,
            "agents": [{"name": a.name, "kind": a.kind,
                        "h_m": a.h_m, "h_r": a.h_r} for a in agents],
            "config": dataclasses.asdict(cfg),
        })

    # ------------------------------------------------------------- перехват

    def attach(self, ce, market, rec, agents: list) -> None:
        """Обернуть ce.submit / market.execute_hedges (v1: quote_hedge) / rec.hedge.

        Симуляция вызывает их как раньше; журнал перехватывает события,
        не меняя ни аргументов, ни результатов.
        """
        orig_submit = ce.submit

        def submit(order):
            oid = orig_submit(order)
            self._orders.append({
                "oid": oid,
                "agent": order.agent,
                "kind": self._kind.get(order.agent),
                "weights": dict(order.weights),
                "z": order.z,
                "lam": order.lam,
                "lam_ccy": order.lam_ccy,
                "expiry": order.expiry,
                "meta": dict(order.meta) if order.meta else {},
            })
            return oid

        ce.submit = submit

        orig_quote = market.quote_hedge

        def quote_hedge(venue, side, size):
            res = orig_quote(venue, side, size)
            self._last_quote = {"venue": venue, "side": side,
                                "requested": size, **res}
            return res

        market.quote_hedge = quote_hedge

        # v2: батч-исполнение хеджей — запоминаем результат по каждому агенту
        # и книгу площадки ДО батча (после него она уже объедена)
        if hasattr(market, "execute_hedges"):
            orig_batch = market.execute_hedges

            def execute_hedges(venue, orders):
                ex = market.exchanges[venue]
                # состояние площадки, по которому агенты принимали решение:
                # батч съедает книгу и двигает last_price (а с ним и mid при
                # пустой стороне), поэтому конца тика недостаточно
                self._pre_books[venue] = {
                    "mid": ex.mid,
                    "last_price": ex.last_price,
                    "spread": ex.spread,
                    "best_bid": ex.best_bid,
                    "best_ask": ex.best_ask,
                    "book": {"bids": _book_levels(ex.bids, best_first_desc=True),
                             "asks": _book_levels(ex.asks, best_first_desc=False)}}
                res = orig_batch(venue, orders)
                for agent, side, size in orders:
                    self._batch[agent] = {"side": side, "requested": size,
                                          **(res.get(agent) or {})}
                return res

            market.execute_hedges = execute_hedges

        # rec.hedge вызывается после исполнения; детали берём из батча по
        # имени агента (v2), иначе — из последнего quote_hedge (v1)
        orig_hedge = rec.hedge

        def hedge(t, venue, agent_idx, notional, result_value):
            orig_hedge(t, venue, agent_idx, notional, result_value)
            q = self._batch.pop(agents[agent_idx].name, None) \
                or self._last_quote or {}
            self._hedges.append({
                "agent": agents[agent_idx].name,
                "venue": venue,
                "side": q.get("side"),            # сторона агента в стакане
                "requested": q.get("requested"),  # запрошенный объём, шт
                "filled": q.get("filled"),        # исполнено, шт
                "avg_price": q.get("avg_price"),  # средняя цена исполнения
                "total_cost": q.get("total_cost"),  # стоимость, кэш площадки
                "notional_mid": notional,         # исполнено * mid площадки
                "value": result_value,            # вклад хеджа в PnL, X1
            })
            self._last_quote = None

        rec.hedge = hedge

    # --------------------------------------------------------------- запись

    def tick(self, t: int, market, ce, agents: list, equity,
             report=None) -> None:
        """Одна строка на тик; report=None только у стартовой записи t=0."""
        venues = {}
        for name in VENUES:
            ex = market.exchanges[name]
            venues[name] = {
                "mid": ex.mid,
                "last_price": ex.last_price,
                "best_bid": ex.best_bid,
                "best_ask": ex.best_ask,
                "spread": ex.spread,
                "depth": ex.depth_near_mid(self._cfg.depth_band),
                "volatility": ex.volatility,
                "auction_volume": ex.last_trade_volume,
                # схема стакана НА КОНЕЦ тика: после реальных хеджей это
                # остаточная книга; состояние до батча — book_before_hedge
                "book": {"bids": _book_levels(ex.bids, best_first_desc=True),
                         "asks": _book_levels(ex.asks, best_first_desc=False)},
            }
            pre = self._pre_books.pop(name, None)
            if pre is not None:
                venues[name]["before_hedge"] = pre

        ce_rec: dict = {"prices": ce.prices()}
        fills: list[dict] = []
        if report is not None:
            ce_rec.update({
                "gross_notional": report.gross_notional,
                "rank": report.rank,
                "max_value_imbalance": report.max_value_imbalance,
                "cap_iterations": report.cap_iterations,
                "net_value_flow": report.net_value_flow,
                "net_quantity_flow": report.net_quantity_flow,
            })
            fills = [{
                "oid": fl.order_id,
                "agent": fl.agent,
                "kind": self._kind.get(fl.agent),
                "f": fl.f,
                "notional": fl.notional,
                "quantities": fl.quantities,
            } for fl in report.fills]

        agent_rows = []
        for i, a in enumerate(agents):
            bal = ce.balances.get(a.name, {})
            leg = (F.TRANSLATOR_HOME[a.kind][1] if a.kind in F.TRANSLATOR_HOME
                   else F.ARB_LONG_LEG[a.kind])
            q_leg = bal.get(leg, 0.0)
            agent_rows.append({
                "name": a.name,
                "kind": a.kind,
                "active": a.active,
                "death_tick": a.death_tick,
                "equity": float(equity[i, t]),      # PnL всего, X1
                "capital": F.capital(float(equity[i, t]), self._cfg.c0),
                "balances": dict(bal),              # штуки каждого актива
                "leg": leg,                         # торгуемая нога
                "inventory_units": q_leg,
                "inventory_value": q_leg * ce.price(leg),
            })

        self._write({
            "type": "tick",
            "t": int(t),
            "world": {"fundamental": market.fundamental,
                      "anchor": market.anchor,
                      "venues": venues},
            "ce": ce_rec,
            "orders": self._orders,
            "fills": fills,
            "hedges": self._hedges,
            "agents": agent_rows,
        })
        self._orders, self._hedges = [], []
        if t % 1000 == 0:
            self._fh.flush()

    def _write(self, record: dict) -> None:
        self._fh.write(json.dumps(record, ensure_ascii=False,
                                  default=_jsonable))
        self._fh.write("\n")

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()
