"""
report_data.py — данные прогона для отчёта: запуск, кэш, фазы, сводки.

Модуль ничего не настраивает: конфигурация эксперимента живёт целиком в
SimConfig (agent_simulation.py), сюда она приходит как есть. Здесь только

    RunData        — плоский контейнер массивов прогона (всё, что рисуется);
    load_or_build  — прогон с кэшем: повторная сборка отчёта без изменений
                     в конфиге и в коде симуляции берёт готовый .npz;
    phases         — деление прогона на прогрев / отбор / стационар;
    summary        — числа для таблиц документа.

Кэш инвалидируется автоматически: ключ считается и по конфигу, и по
содержимому исходников симуляции, поэтому правка любой формулы приводит
к честному пересчёту.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np

import agent_formulas as F
import agent_simulation as A
from coupled_market import ExchangeConfig

# исходники симуляции: их правка меняет результат прогона и обнуляет кэш
# (модули отчёта сюда не входят — они только читают готовые массивы)
_SOURCES = ("agent_simulation.py", "coupled_market.py",
            "pfx_exchange.py", "agent_formulas.py")

KINDS = A.KINDS
VENUES = A.VENUES

KIND_LABEL = {
    "T1": "T1 — трансляторы биржи 1",
    "T2": "T2 — трансляторы биржи 2",
    "AX": "AX — арбитражёры X1/X2",
    "AY": "AY — арбитражёры Y1/Y2",
}
VENUE_LABEL = {"1": "биржа 1 (толстая)", "2": "биржа 2 (тонкая)"}


# --------------------------------------------------------------------------- #
# Контейнер прогона
# --------------------------------------------------------------------------- #

@dataclass
class RunData:
    """Всё, что нужно отчёту, в виде массивов numpy.

    Время везде — индекс тика t = 0..T; нулевой тик это состояние после
    прогрева лимитных бирж и до первого клиринга CE.
    """

    # --- скалярное и структурное ------------------------------------------ #
    cfg: A.SimConfig
    gamma: float
    stats: dict
    agents: list                  # [{name, kind, h_m, h_r, death_tick}, ...]
    deaths: list                  # [(тик, имя, убыток за окно), ...]
    elapsed: float                # длительность прогона, с

    # --- агенты, (n, T+1) --------------------------------------------------- #
    equity: np.ndarray            # PnL, X1
    inventory: np.ndarray         # стоимость позиции по торгуемой ноге, X1
    turnover: np.ndarray          # |нотионал| сделок на CE за тик, X1
    reval: np.ndarray             # компонента PnL: переоценка позиции
    hedge_pnl: np.ndarray         # компонента PnL: результат хеджа

    # --- мир, (T+1,) и (T+1, 2) -------------------------------------------- #
    fundamental: np.ndarray       # справедливая (экзогенная) цена
    anchor: np.ndarray            # якорь генерации фонового потока
    mid: np.ndarray
    last_price: np.ndarray
    best_bid: np.ndarray
    best_ask: np.ndarray
    spread: np.ndarray
    depth: np.ndarray             # объём в полосе +-depth_band от мида
    vol: np.ndarray               # EWMA-волатильность мида за тик
    volume: np.ndarray            # объём аукциона за тик

    # --- непрерывная биржа --------------------------------------------------#
    ce_prices: np.ndarray         # (T+1, 4) в порядке F.ASSETS
    ce_gross: np.ndarray          # оборот за тик, X1
    ce_gross_kind: np.ndarray     # (T+1, 4) оборот по типам агентов
    ce_absf_kind: np.ndarray      # (T+1, 4) среднее |f| по типам
    ce_orders: np.ndarray         # число исполнившихся заявок за тик
    ce_rank: np.ndarray           # ранг книги
    ce_imbalance: np.ndarray      # невязка клиринга
    pnl_residual: np.ndarray      # невязка разложения PnL

    # --- хедж, (T+1, 2) ------------------------------------------------------#
    hedge_count: np.ndarray
    hedge_notional: np.ndarray
    hedge_result: np.ndarray      # выигрыш(+)/потеря(-) на хедже, X1

    # --- срезы стакана ------------------------------------------------------ #
    book_ticks: np.ndarray        # (S,) тики срезов
    book_bid: np.ndarray          # (S, 2, B) объём покупок по бинам
    book_ask: np.ndarray          # (S, 2, B) объём продаж по бинам
    book_edges: np.ndarray        # (B+1,) границы бинов, смещение от мида

    # ------------------------------------------------------------------ виды

    @property
    def T(self) -> int:
        return self.equity.shape[1] - 1

    @property
    def n(self) -> int:
        return self.equity.shape[0]

    @property
    def kinds(self) -> np.ndarray:
        """Тип каждого агента как строковый массив."""
        return np.array([a["kind"] for a in self.agents])

    def kind_rows(self, kind: str) -> np.ndarray:
        """Индексы агентов заданного типа."""
        return np.array([i for i, a in enumerate(self.agents)
                         if a["kind"] == kind], dtype=int)

    def hyper(self, key: str) -> np.ndarray:
        """Значение гиперпараметра (h_m или h_r) по агентам; nan где нет."""
        return np.array([np.nan if a[key] is None else a[key]
                         for a in self.agents], dtype=float)

    def alive_mask(self) -> np.ndarray:
        """(n, T+1) bool: агент ещё торговал на этом тике."""
        T = self.T
        death = np.array([T + 1 if a["death_tick"] is None else a["death_tick"]
                          for a in self.agents], dtype=float)
        return np.arange(T + 1)[None, :] <= death[:, None]

    def price(self, asset: str) -> np.ndarray:
        """Ряд цены актива на CE."""
        return self.ce_prices[:, F.ASSETS.index(asset)]

    # -------------------------------------------------------------- хранение

    def save(self, path: Path) -> None:
        arrays = {f.name: getattr(self, f.name) for f in fields(self)
                  if isinstance(getattr(self, f.name), np.ndarray)}
        meta = {"cfg": asdict(self.cfg), "gamma": self.gamma,
                "stats": self.stats, "agents": self.agents,
                "deaths": self.deaths, "elapsed": self.elapsed}
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, _meta=np.array(json.dumps(meta)), **arrays)

    @classmethod
    def load(cls, path: Path) -> "RunData":
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["_meta"]))
            arrays = {k: z[k] for k in z.files if k != "_meta"}
        return cls(cfg=cfg_from_dict(meta["cfg"]), gamma=meta["gamma"],
                   stats=meta["stats"], agents=meta["agents"],
                   deaths=[tuple(d) for d in meta["deaths"]],
                   elapsed=meta["elapsed"], **arrays)


def cfg_from_dict(d: dict) -> A.SimConfig:
    """Обратное преобразование дампа конфига (вложенные ExchangeConfig)."""
    d = dict(d)
    d["venue1"] = ExchangeConfig(**d["venue1"])
    d["venue2"] = ExchangeConfig(**d["venue2"])
    for key in ("h_m_values", "h_r_values"):
        d[key] = tuple(d[key])
    return A.SimConfig(**d)


# --------------------------------------------------------------------------- #
# Прогон и кэш
# --------------------------------------------------------------------------- #

def _cache_key(cfg: A.SimConfig) -> str:
    """Хеш конфига и исходников симуляции."""
    h = hashlib.sha1(json.dumps(asdict(cfg), sort_keys=True).encode())
    here = Path(__file__).parent
    for name in _SOURCES:
        src = here / name
        if src.exists():
            h.update(src.read_bytes())
    return h.hexdigest()[:16]


def build(cfg: A.SimConfig, verbose: bool = True) -> RunData:
    """Полный прогон симуляции с записью всех рядов."""
    t0 = time.time()
    res = A.run_simulation(cfg, verbose=verbose)
    rec = res.rec
    agents = [{"name": a.name, "kind": a.kind, "h_m": a.h_m, "h_r": a.h_r,
               "death_tick": a.death_tick} for a in res.agents]
    return RunData(
        cfg=cfg, gamma=res.gamma, stats=dict(res.stats), agents=agents,
        deaths=[(int(t), str(name), float(loss))
                for t, name, loss in res.deaths],
        elapsed=time.time() - t0,
        equity=res.equity, inventory=res.inventory, turnover=rec.turnover,
        reval=rec.reval, hedge_pnl=rec.hedge_pnl,
        fundamental=rec.fundamental, anchor=rec.anchor, mid=rec.mid,
        last_price=rec.last_price, best_bid=rec.best_bid,
        best_ask=rec.best_ask, spread=rec.spread, depth=rec.depth,
        vol=rec.vol, volume=rec.volume,
        ce_prices=res.ce_prices, ce_gross=rec.ce_gross,
        ce_gross_kind=rec.ce_gross_kind, ce_absf_kind=rec.ce_absf_kind,
        ce_orders=rec.ce_orders, ce_rank=rec.ce_rank,
        ce_imbalance=rec.ce_imbalance, pnl_residual=rec.pnl_residual,
        hedge_count=rec.hedge_count, hedge_notional=rec.hedge_notional,
        hedge_result=rec.hedge_result,
        book_ticks=rec.book_ticks, book_bid=rec.book_bid,
        book_ask=rec.book_ask, book_edges=rec.book_edges)


def load_or_build(cfg: A.SimConfig | None = None,
                  verbose: bool = True) -> RunData:
    """Прогон из кэша, а при изменении конфига или кода — заново."""
    cfg = cfg or A.SimConfig()
    cache = Path(cfg.report_dir) / "cache" / f"run-{_cache_key(cfg)}.npz"
    if cache.exists():
        if verbose:
            print(f"Прогон взят из кэша: {cache}")
        return RunData.load(cache)
    if verbose:
        print(f"Кэша нет, запускаю симуляцию: {cfg.total_steps} тиков")
    run = build(cfg, verbose=verbose)
    run.save(cache)
    if verbose:
        print(f"Прогон записан в кэш: {cache} "
              f"({cache.stat().st_size / 2 ** 20:.1f} МБ, "
              f"{run.elapsed:.1f} с)")
    return run


# --------------------------------------------------------------------------- #
# Фазы прогона
# --------------------------------------------------------------------------- #

@dataclass
class Phases:
    """Три фазы прогона.

    прогрев   — до включения отбора: живы все 64 агента, гиперпараметры
                никак не отобраны, цены CE ещё «нащупывают» рынок;
    отбор     — от evolution_start до последнего выбытия: состав популяции
                меняется, доли мощности перетекают выжившим;
    стационар — после последнего выбытия: состав зафиксирован, режим
                установившийся. Именно его честно сравнивать с прогревом.
    """

    warmup: tuple
    selection: tuple
    stationary: tuple
    note: str

    @property
    def named(self) -> list:
        return [("прогрев", self.warmup), ("отбор", self.selection),
                ("стационар", self.stationary)]


def phases(run: RunData) -> Phases:
    T, ev = run.T, min(run.cfg.evolution_start, run.T)
    last_death = max((t for t, _, _ in run.deaths), default=None)
    if last_death is None:
        return Phases((1, ev), (ev, ev), (ev, T),
                      "выбытий не было: стационаром считается всё после "
                      "включения отбора")
    if T - last_death < 0.05 * T:
        start = int(0.75 * T)
        return Phases((1, ev), (ev, last_death), (start, T),
                      f"последнее выбытие на тике {last_death} — слишком "
                      f"близко к концу прогона, поэтому за стационар взята "
                      f"последняя четверть ({start}..{T})")
    return Phases((1, ev), (ev, last_death), (last_death, T),
                  f"последнее выбытие на тике {last_death}; дальше состав "
                  f"популяции не меняется")


def window_slice(phase: tuple) -> slice:
    return slice(phase[0], phase[1] + 1)


# --------------------------------------------------------------------------- #
# Производные ряды
# --------------------------------------------------------------------------- #

def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """Скользящее среднее с сохранением длины (край усредняется частично)."""
    window = max(int(window), 1)
    if window <= 1:
        return np.asarray(x, dtype=float)
    x = np.asarray(x, dtype=float)
    filled = np.nan_to_num(x)
    valid = (~np.isnan(x)).astype(float)
    kernel = np.ones(window)
    num = np.convolve(filled, kernel, mode="full")[:len(x)]
    den = np.convolve(valid, kernel, mode="full")[:len(x)]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def translation_error_bp(run: RunData) -> np.ndarray:
    """(T+1, 2) ошибка трансляции цены на CE, базисные пункты.

    Транслятор T1 котирует Y1 около мида биржи 1, T2 — Y2 около мида биржи 2,
    но цены Y2 и X2 живут в единицах X2, поэтому сравнивать надо
    P_Y2 / P_X2 с мидом второй биржи.
    """
    y1 = run.price("Y1")
    y2 = run.price("Y2") / run.price("X2")
    out = np.column_stack([y1 / run.mid[:, 0] - 1.0,
                           y2 / run.mid[:, 1] - 1.0])
    return out * 1e4


def basis_bp(run: RunData) -> np.ndarray:
    """(T+1, 2) базисы арбитражёров в базисных пунктах.

    AX торгует X1 против X2 с безразличием при курсе 1, AY — Y1 против Y2.
    Оба базиса — это лог-расхождение f, на которое реагирует заявка.
    """
    x = run.price("X1") / run.price("X2") - 1.0
    y = run.price("Y1") / run.price("Y2") - 1.0
    return np.column_stack([x, y]) * 1e4


def hyper_shares(run: RunData, kind: str, key: str) -> tuple:
    """Доли значений гиперпараметра среди живых агентов типа во времени.

    Возвращает (значения, (V, T+1) доли) — сумма долей по значениям равна 1,
    пока в типе кто-то жив.
    """
    rows = run.kind_rows(kind)
    values = np.array(sorted({run.agents[i][key] for i in rows
                              if run.agents[i][key] is not None}))
    alive = run.alive_mask()[rows]
    counts = np.zeros((len(values), run.T + 1))
    for v, value in enumerate(values):
        mask = np.array([run.agents[i][key] == value for i in rows])
        counts[v] = alive[mask].sum(axis=0)
    total = counts.sum(axis=0)
    shares = np.divide(counts, total, out=np.zeros_like(counts),
                       where=total > 0)
    return values, shares


def phase_pnl(run: RunData, phase: tuple) -> np.ndarray:
    """Прирост PnL каждого агента за фазу."""
    return run.equity[:, phase[1]] - run.equity[:, phase[0]]


def phase_mean(x: np.ndarray, phase: tuple, axis: int = -1) -> np.ndarray:
    """Среднее по времени внутри фазы (последняя ось — время)."""
    sl = window_slice(phase)
    if x.ndim == 1:
        return float(np.nanmean(x[sl]))
    return np.nanmean(x[..., sl] if axis == -1 else x[sl], axis=axis)


# --------------------------------------------------------------------------- #
# Сводка для таблиц документа
# --------------------------------------------------------------------------- #

def summary(run: RunData) -> dict:
    """Числа прогона, которые печатаются в таблицах отчёта."""
    ph = phases(run)
    T = run.T
    err = translation_error_bp(run)
    bas = basis_bp(run)
    alive = run.alive_mask()

    world = {}
    for v, name in enumerate(VENUES):
        sl = window_slice((1, T))
        spread = run.spread[sl, v]
        world[name] = {
            "spread_mean": float(np.nanmean(spread)),
            "spread_p90": float(np.nanpercentile(spread[~np.isnan(spread)], 90)),
            "oneside_share": float(np.mean(np.isnan(spread))),
            "depth_mean": float(np.mean(run.depth[sl, v])),
            "vol_mean": float(np.mean(run.vol[sl, v])),
            "volume_mean": float(np.mean(run.volume[sl, v])),
            "trade_share": float(np.mean(run.volume[sl, v] > 0)),
            "mid_last": float(run.mid[T, v]),
            "mid_vs_fund_bp": float(np.mean(
                (run.mid[sl, v] / run.fundamental[sl] - 1.0) * 1e4)),
            "hedge_count": float(run.hedge_count[:, v].sum()),
            "hedge_notional": float(run.hedge_notional[:, v].sum()),
            "hedge_result": float(run.hedge_result[:, v].sum()),
        }

    ce = {"gross_mean": float(run.ce_gross[1:].mean()),
          "gross_total": float(run.ce_gross.sum()),
          "orders_mean": float(run.ce_orders[1:].mean()),
          "rank_mode": int(np.bincount(run.ce_rank[1:].astype(int)).argmax()),
          "imbalance_max": float(run.ce_imbalance.max()),
          "pnl_residual_max": float(run.pnl_residual.max())}
    for k, kind in enumerate(KINDS):
        ce[f"gross_{kind}"] = float(run.ce_gross_kind[:, k].sum())

    by_kind = {}
    for kind in KINDS:
        rows = run.kind_rows(kind)
        by_kind[kind] = {
            "n": len(rows),
            "alive": int(alive[rows, T].sum()),
            "pnl_total": float(run.equity[rows, T].sum()),
            "pnl_mean": float(run.equity[rows, T].mean()),
            "pnl_best": float(run.equity[rows, T].max()),
            "pnl_worst": float(run.equity[rows, T].min()),
            "pnl_warmup": float(phase_pnl(run, ph.warmup)[rows].sum()),
            "pnl_stationary": float(phase_pnl(run, ph.stationary)[rows].sum()),
            "turnover": float(run.turnover[rows].sum()),
            "reval": float(run.reval[rows].sum()),
            "hedge": float(run.hedge_pnl[rows].sum()),
            "inventory_abs": float(np.abs(run.inventory[rows]).mean()),
        }

    phase_stats = {}
    for label, ph_range in ph.named:
        sl = window_slice(ph_range)
        phase_stats[label] = {
            "range": ph_range,
            "err1_mean": float(np.nanmean(err[sl, 0])),
            "err1_std": float(np.nanstd(err[sl, 0])),
            "err2_mean": float(np.nanmean(err[sl, 1])),
            "err2_std": float(np.nanstd(err[sl, 1])),
            "basis_x_std": float(np.nanstd(bas[sl, 0])),
            "basis_y_std": float(np.nanstd(bas[sl, 1])),
            "gross_mean": float(np.mean(run.ce_gross[sl])),
        }

    return {"world": world, "ce": ce, "kinds": by_kind, "phases": phase_stats,
            "phases_obj": ph, "gamma": run.gamma, "elapsed": run.elapsed,
            "n_agents": run.n, "alive": int(alive[:, T].sum()),
            "deaths": len(run.deaths)}


if __name__ == "__main__":
    run = load_or_build()
    ph = phases(run)
    print("фазы:", ph.named, "\n", ph.note)
    s = summary(run)
    print("живых:", s["alive"], "из", s["n_agents"], " gamma:", round(s["gamma"], 3))
    print("оборот CE/тик:", round(s["ce"]["gross_mean"], 4))
    for kind in KINDS:
        k = s["kinds"][kind]
        print(f"  {kind}: живых {k['alive']}/{k['n']}, PnL {k['pnl_total']:+.3f}, "
              f"переоценка {k['reval']:+.3f}, хедж {k['hedge']:+.3f}")
