"""
pfx_exchange.py — continuous exchange with portfolio orders, cleared in log-price space.

MODEL
-----
The exchange lists n assets.  Currencies and cash balances are assets like any
other; there is no privileged "money" asset.

Internal state is the vector of log-prices  S in R^n.  Actual prices are
P = exp(S), and only ratios of prices are determined by the book — the overall
level is fixed by an explicit gauge (see `unit_of_account`).

An order is the triple (w, z, lam):

    w    weights by VALUE, one entry per leg, with  sum_i w_i = 0
         (self-financing: value in == value out)
    z    indifference rate: the level of the geometric index
             Pi(P) = prod_i P_i^{w_i}
         at which the order trades nothing.  Because sum w_i = 0, Pi is a pure
         ratio of prices and z does not depend on any choice of numeraire.
    lam  aggression: money per unit time per unit of log-mispricing.

The order's signal is the log-mispricing

    f = log(z) - w . S            (dimensionless, ~ relative cheapness)

and it trades notional at rate  lam * f  (positive f => buys the basket).

CLEARING
--------
Net value flow into asset i must vanish:

    sum_a lam_a f_a w_{a,i} = 0   for every i

Substituting f_a gives a linear system in S:

    A S = b,    A = sum_a lam_a w_a w_a^T  (PSD),   b = sum_a lam_a log(z_a) w_a

equivalently the lam-weighted least squares problem

    S* = argmin_S  sum_a lam_a ( w_a . S - log z_a )^2

Because sum_i w_{a,i} = 0 for every order, A @ 1 = 0: the solution is unique
only up to S -> S + c*1, i.e. up to a common rescaling of all prices.  That
null direction is exactly the choice of numeraire.  More generally, ker(A) is
the set of price directions no order responds to, which also covers any asset
nobody quotes this step.

The system is solved in INCREMENT form, which makes all of that exact:

    r_prev = b - A S_prev          net value flow at the incoming prices
    delta  = A^+ r_prev            minimum-norm correction
    S      = S_prev + delta + c*1  c restores the gauge

Three facts make this leak-free:

  * r_prev = W^T Lam f_prev lies in range(A) by construction, so A A^+ r_prev
    = r_prev and the clearing residual b - A S is zero to machine precision;
  * the minimum-norm delta has no component in ker(A), so directions nobody
    trades — including unquoted assets — keep their previous log-price exactly,
    with no regularisation parameter to tune;
  * W @ 1 = 0, so the gauge shift c*1 is invisible to every order and cannot
    disturb the flows.

UNITS
-----
    w        dimensionless (value shares)
    z        price ratio, defined by the shape of w
    f        dimensionless
    lam      [money] / [time]        <- the only place money enters an order
    balances physical quantities of each asset
    P&L      quantities x prices, in whatever asset you choose to report in

"money" above means the exchange's unit of account, which is the same object
as the gauge: pinning pi . S = g both fixes the price level and declares what
one unit of account is.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np

__all__ = ["Order", "Fill", "ClearingReport", "Exchange"]


# --------------------------------------------------------------------------- #
# Order specification (immutable; what a participant hands to the exchange)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Order:
    """A resting portfolio order.

    Core fields
    -----------
    weights : value shares per asset, must sum to 0 (validated on submit).
              {"X1": +1, "RUB": -1}       buy X1 with cash
              {"X1": +1, "X2": -1}        venue basis, no cash leg
              {"A": +0.6, "B": +0.4, "USD": -1}   buy a 60/40 basket for dollars
    z       : indifference level of the geometric index prod P_i^{w_i}, > 0.
              For {"X1": +1, "RUB": -1} this is simply the price of X1 in RUB.
    lam     : notional per unit time per unit log-mispricing, >= 0.

    Optional fields (all default to "off"; add more here to extend)
    ---------------------------------------------------------------
    agent        : owner, used to route fills into balances.
    lam_ccy      : if set, `lam` is quoted in that asset's units instead of the
                   unit of account.  The engine converts using the PREVIOUS
                   step's price (an explicit one-step lag, which keeps clearing
                   an exact linear solve).
    max_rate     : cap on |notional per unit time| for this order.
    max_notional : lifetime budget on cumulative |notional|; the order
                   deactivates once spent.
    expiry       : exchange time after which the order is inactive.
    meta         : free-form payload, never touched by the engine.
    """

    weights: Mapping[str, float]
    z: float
    lam: float
    agent: str = "anon"
    lam_ccy: str | None = None
    max_rate: float | None = None
    max_notional: float | None = None
    expiry: float | None = None
    meta: dict = field(default_factory=dict)


@dataclass
class _LiveOrder:
    """Exchange-side mutable state attached to a submitted Order."""
    oid: int
    spec: Order
    w: np.ndarray          # dense weight vector over the asset universe
    log_z: float
    active: bool = True
    spent: float = 0.0     # cumulative |notional| traded


# --------------------------------------------------------------------------- #
# Clearing output
# --------------------------------------------------------------------------- #

@dataclass
class Fill:
    """What one order traded during one clearing step."""
    order_id: int
    agent: str
    f: float                        # log-mispricing at the clearing price
    notional: float                 # signed value traded, in unit of account
    quantities: dict[str, float]    # signed quantity delta per asset


@dataclass
class ClearingReport:
    t: float
    dt: float
    prices: dict[str, float]
    log_prices: dict[str, float]
    fills: list[Fill]
    net_quantity_flow: dict[str, float]   # ~0 for every asset
    net_value_flow: dict[str, float]      # residual of the clearing condition
    gross_notional: float                 # sum of |notional| over fills
    max_value_imbalance: float            # clearing residual; ~machine zero
                                          # unless the book is near-degenerate
    rank: int                             # number of price directions the book
                                          # actually determines this step
    cap_iterations: int                   # rounds of lambda tightening used


# --------------------------------------------------------------------------- #
# Exchange
# --------------------------------------------------------------------------- #

class Exchange:
    """Continuous exchange for portfolio orders.

    Prices in, prices out, everywhere on the public surface.  Log-prices exist
    only between `_solve` and the fill computation.
    """

    _MAX_CAP_ITERS = 8

    def __init__(
        self,
        assets: Sequence[str],
        prices: Mapping[str, float],
        unit_of_account: str | Mapping[str, float] = None,
        unit_price: float = 1.0,
        rank_tol: float = 1e-12,
        weight_tol: float = 1e-9,
    ):
        """
        assets          : the universe, cash/currencies included.
        prices          : starting price of every asset, in the unit of account.
        unit_of_account : asset name, or a basket {asset: weight}, whose price is
                          pinned to `unit_price`.  This simultaneously (a) fixes
                          the otherwise-free overall price level and (b) declares
                          the money unit in which `lam`, notionals and marks are
                          expressed.  Defaults to the first asset.
        unit_price      : the pinned price, normally 1.0.
        rank_tol        : relative cutoff on the eigenvalues of A used to decide
                          which price directions the book determines.  A
                          direction below the cutoff is treated as untraded and
                          keeps its previous price.  This is a rank decision,
                          not an error knob: directions above the cutoff are
                          solved exactly.
        weight_tol      : tolerance for the sum(w) == 0 check.
        """
        if len(set(assets)) != len(assets):
            raise ValueError("duplicate asset names")
        self.assets = list(assets)
        self.index = {a: i for i, a in enumerate(self.assets)}
        self.n = len(self.assets)
        self.rank_tol = float(rank_tol)
        self.weight_tol = float(weight_tol)

        missing = set(self.assets) - set(prices)
        if missing:
            raise ValueError(f"no starting price for {sorted(missing)}")
        if any(prices[a] <= 0 for a in self.assets):
            raise ValueError("all prices must be strictly positive")

        # --- gauge vector pi and target level g ----------------------------- #
        if unit_of_account is None:
            unit_of_account = self.assets[0]
        pi = np.zeros(self.n)
        if isinstance(unit_of_account, str):
            pi[self._i(unit_of_account)] = 1.0
        else:
            for a, x in unit_of_account.items():
                pi[self._i(a)] = float(x)
        if abs(pi.sum()) < 1e-12:
            # 1 is always in ker(A), so pi must have a component along it;
            # otherwise the gauge cannot pin the price level at all, and the
            # projection used in _solve would be degenerate.
            raise ValueError("unit_of_account weights must not sum to zero")
        self._pi = pi
        self._g = math.log(unit_price)

        # --- state ----------------------------------------------------------#
        self.S = np.array([math.log(prices[a]) for a in self.assets])
        self._normalize_gauge()

        self.t = 0.0
        self._next_oid = 1
        self._live: dict[int, _LiveOrder] = {}
        self.balances: dict[str, dict[str, float]] = {}

    # ---------------------------------------------------------------- helpers

    def _i(self, asset: str) -> int:
        try:
            return self.index[asset]
        except KeyError:
            raise ValueError(f"unknown asset {asset!r}") from None

    def _normalize_gauge(self) -> None:
        """Shift all log-prices by a constant so that pi . S == g.

        This is a pure change of money unit: every price ratio is untouched.
        """
        c = (self._g - self._pi @ self.S) / self._pi.sum()
        self.S = self.S + c

    def _dense(self, weights: Mapping[str, float]) -> np.ndarray:
        w = np.zeros(self.n)
        for a, x in weights.items():
            w[self._i(a)] = float(x)
        return w

    # ------------------------------------------------------------ public read

    def price(self, asset: str) -> float:
        return math.exp(self.S[self._i(asset)])

    def prices(self) -> dict[str, float]:
        return {a: math.exp(s) for a, s in zip(self.assets, self.S)}

    def unit_of_account_price(self) -> float:
        """Current price of the gauge asset/basket.

        Equals the `unit_price` passed at construction, always.  Useful as a
        cheap assertion that the gauge constraint is being honoured.
        """
        return math.exp(self._pi @ self.S)

    def rate(self, weights: Mapping[str, float]) -> float:
        """Current level of the geometric index prod P_i^{w_i}.

        For weights summing to zero this is a pure exchange rate, directly
        comparable with an order's `z`.
        """
        return math.exp(self._dense(weights) @ self.S)

    def book(self) -> list[Order]:
        return [lo.spec for lo in self._live.values() if lo.active]

    def mark_to_market(self, agent: str) -> float:
        """Value of an agent's holdings, in the unit of account."""
        bal = self.balances.get(agent, {})
        return sum(q * self.price(a) for a, q in bal.items())

    # ----------------------------------------------------------- order entry

    def submit(self, order: Order) -> int:
        """Validate and register an order.  Returns its id."""
        w = self._dense(order.weights)
        if abs(w.sum()) > self.weight_tol:
            raise ValueError(
                f"weights must sum to 0 (self-financing), got {w.sum():.3e}. "
                "Include the cash leg as an asset."
            )
        if not np.any(w):
            raise ValueError("all weights are zero")
        if order.z <= 0:
            raise ValueError("z must be strictly positive")
        if order.lam < 0:
            raise ValueError("lam must be non-negative")
        if order.lam_ccy is not None:
            self._i(order.lam_ccy)          # existence check

        oid = self._next_oid
        self._next_oid += 1
        self._live[oid] = _LiveOrder(oid=oid, spec=order, w=w, log_z=math.log(order.z))
        self.balances.setdefault(order.agent, {})
        return oid

    def cancel(self, oid: int) -> None:
        lo = self._live.get(oid)
        if lo is not None:
            lo.active = False

    def deposit(self, agent: str, asset: str, quantity: float) -> None:
        """Credit an agent's balance (funding, transfers in from other venues)."""
        self._i(asset)
        self.balances.setdefault(agent, {})
        self.balances[agent][asset] = self.balances[agent].get(asset, 0.0) + quantity

    # -------------------------------------------------------------- clearing

    def _select(self, dt: float) -> tuple[list[_LiveOrder], np.ndarray]:
        """Active orders and their base lambdas, in unit-of-account per time.

        Expiries are resolved here, and `lam_ccy` is converted using the
        PREVIOUS log-prices — an explicit one-step lag that keeps the clearing
        solve exactly linear in S.
        """
        orders: list[_LiveOrder] = []
        lams: list[float] = []

        for lo in self._live.values():
            if not lo.active:
                continue
            spec = lo.spec
            if spec.expiry is not None and self.t >= spec.expiry:
                lo.active = False
                continue
            if spec.max_notional is not None and lo.spent >= spec.max_notional:
                lo.active = False
                continue

            lam = spec.lam
            if spec.lam_ccy is not None:
                lam *= math.exp(self.S[self._i(spec.lam_ccy)])
            if lam <= 0:
                continue

            orders.append(lo)
            lams.append(lam)

        return orders, np.asarray(lams, dtype=float)

    def _tighten(self, orders: list[_LiveOrder], lams: np.ndarray,
                 S: np.ndarray, dt: float) -> tuple[np.ndarray, bool]:
        """Reduce lambdas so that rate and budget limits hold at prices `S`.

        Lambdas only ever go down, so repeated application converges.  Applying
        the caps BEFORE the solve (rather than truncating fills afterwards) is
        what keeps the clearing condition exact: a truncated fill would leave
        the book unbalanced.
        """
        out = lams.copy()
        changed = False
        for k, lo in enumerate(orders):
            spec = lo.spec
            af = abs(lo.log_z - lo.w @ S)
            if af <= 0:
                continue
            cap = out[k]
            if spec.max_rate is not None:
                cap = min(cap, spec.max_rate / af)
            if spec.max_notional is not None:
                remaining = max(spec.max_notional - lo.spent, 0.0)
                cap = min(cap, remaining / (af * dt))
            if cap < out[k] * (1.0 - 1e-12):
                out[k] = cap
                changed = True
        return out, changed

    def _solve(self, orders: list[_LiveOrder],
               lams: np.ndarray) -> tuple[np.ndarray, int]:
        """Clear the book.  Returns the new log-prices and the book's rank.

        Solved as an increment from the incoming prices:

            r_prev = b - A S_prev      net value flow at the old prices
            delta  = A^+ r_prev        minimum-norm correction
            S      = S_prev + delta    then shifted to restore the gauge

        A is symmetric PSD, so its eigendecomposition gives A^+ directly.
        Eigen-directions below `rank_tol` are the ones no order responds to;
        delta is zero there, which is exactly "keep the previous price" and
        needs no regularisation term.  Since r_prev lies in range(A) by
        construction, discarding those directions costs nothing.
        """
        n = self.n
        A = np.zeros((n, n))
        r_prev = np.zeros(n)
        for lo, lam in zip(orders, lams):
            A += lam * np.outer(lo.w, lo.w)
            # value flow this order would push at the incoming prices
            r_prev += lam * (lo.log_z - lo.w @ self.S) * lo.w

        # eigh is exact for symmetric matrices and returns ascending values
        vals, vecs = np.linalg.eigh(A)
        cutoff = self.rank_tol * max(vals[-1], 0.0) if n else 0.0
        keep = vals > cutoff
        rank = int(np.count_nonzero(keep))

        delta = np.zeros(n)
        if rank:
            V = vecs[:, keep]
            delta = V @ ((V.T @ r_prev) / vals[keep])

        # `delta` above solves the flows, but its component inside ker(A) is an
        # arbitrary Euclidean artefact: it can move an untraded asset relative
        # to the numeraire even though no order referenced either.  We are free
        # to add anything from ker(A), since no order can feel it, so add the
        # smallest such correction that restores the gauge pi . S = g.
        #
        # That correction points along the projection of pi onto ker(A), which
        # automatically leaves untraded directions alone.  When every asset is
        # quoted, ker(A) = span(1) and this reduces to a plain rescaling of all
        # prices.
        K = vecs[:, ~keep]
        if K.shape[1]:
            q = K @ (K.T @ self._pi)                 # projection of pi onto ker
            denom = float(self._pi @ q)              # = ||K^T pi||^2 > 0
            delta = delta - (float(self._pi @ delta) / denom) * q

        S_new = self.S + delta
        return S_new, rank

    def step(self, dt: float = 1.0) -> ClearingReport:
        """Advance the exchange by dt: clear, fill, settle."""
        if dt <= 0:
            raise ValueError("dt must be positive")

        orders, lams = self._select(dt)
        # First pass uses the incoming prices; then re-tighten against the
        # prices the book actually clears at, until the caps stop binding.
        lams, _ = self._tighten(orders, lams, self.S, dt)
        iters = 0
        for iters in range(1, self._MAX_CAP_ITERS + 1):
            S_new, rank = self._solve(orders, lams)
            lams, changed = self._tighten(orders, lams, S_new, dt)
            if not changed:
                break
        else:
            S_new, rank = self._solve(orders, lams)
        P = np.exp(S_new)

        fills: list[Fill] = []
        gross = 0.0
        value_flow = np.zeros(self.n)     # net value into each asset (should be ~0)
        qty_flow = np.zeros(self.n)

        for lo, lam in zip(orders, lams):
            f = lo.log_z - lo.w @ S_new           # log-mispricing at the clearing price
            V = lam * f * dt                      # notional traded, unit of account
            if V == 0.0:
                continue

            legs = V * lo.w                       # value into each asset
            dq = legs / P                         # -> physical quantities
            value_flow += legs
            qty_flow += dq

            bal = self.balances.setdefault(lo.spec.agent, {})
            qmap: dict[str, float] = {}
            for i, a in enumerate(self.assets):
                if dq[i] != 0.0:
                    bal[a] = bal.get(a, 0.0) + float(dq[i])
                    qmap[a] = float(dq[i])

            gross += abs(V)
            lo.spent += abs(V)
            if lo.spec.max_notional is not None and lo.spent >= lo.spec.max_notional:
                lo.active = False

            fills.append(Fill(order_id=lo.oid, agent=lo.spec.agent, f=f,
                              notional=V, quantities=qmap))

        self.S = S_new
        self.t += dt

        # Drop dead orders from the book.  _select walks every entry in
        # _live, so keeping cancelled/expired/spent orders around would make
        # each step O(all orders ever submitted) in time and memory.
        dead = [oid for oid, lo in self._live.items()
                if not lo.active
                or (lo.spec.expiry is not None and self.t >= lo.spec.expiry)]
        for oid in dead:
            del self._live[oid]

        return ClearingReport(
            t=self.t,
            dt=dt,
            prices={a: float(p) for a, p in zip(self.assets, P)},
            log_prices={a: float(s) for a, s in zip(self.assets, S_new)},
            fills=fills,
            net_quantity_flow={a: float(q) for a, q in zip(self.assets, qty_flow)},
            net_value_flow={a: float(v) for a, v in zip(self.assets, value_flow)},
            gross_notional=gross,
            max_value_imbalance=float(np.max(np.abs(value_flow))) if self.n else 0.0,
            rank=rank,
            cap_iterations=iters,
        )


# --------------------------------------------------------------------------- #
# Demonstration
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    # X1 and X2 are the same underlying listed on two different venues.
    ex = Exchange(
        assets=["RUB", "USD", "X1", "X2"],
        prices={"RUB": 1.0, "USD": 90.0, "X1": 100.0, "X2": 100.0},
        unit_of_account="RUB",
    )
 
    ex.submit(Order(weights={"X1": +1, "RUB": -1}, z=105.0, lam=2e6, agent="buyer"))
    ex.submit(Order(weights={"X1": +1, "RUB": -1}, z=95.0,  lam=2e6, agent="seller"))
    ex.submit(Order(weights={"X1": +1, "X2": -1}, z=1.0,    lam=5e6, agent="arb"))
 
    rep = ex.step(dt=1.0)
 
    print("prices           ", {k: round(v, 4) for k, v in rep.prices.items()})
    print("expected X1      ", round(math.sqrt(105 * 95), 4),
          "  <- geometric mean of the two quotes, equal lambdas")
    print("expected X2      ", "same as X1 (single basis order => zero mispricing)")
    print("USD              ", round(rep.prices["USD"], 4),
          "  <- unquoted, so its price is untouched")
    print("gross notional   ", f"{rep.gross_notional:,.2f}")
    print("book rank        ", rep.rank, " of", len(ex.assets),
          " <- price directions the book determines")
    print("max imbalance    ", f"{rep.max_value_imbalance:.3e}"
          f"  ({rep.max_value_imbalance / rep.gross_notional:.1e} of gross)")
    print()
 
    for fl in rep.fills:
        print(f"  {fl.agent:<7} f={fl.f:+.5f}  notional={fl.notional:+12.2f}  "
              + " ".join(f"{a}{q:+.3f}" for a, q in fl.quantities.items()))
    print()
 
    # Cross rates are read off directly, for any combination of legs.
    print("X1/X2 rate       ", round(ex.rate({"X1": +1, "X2": -1}), 6))
    print("X1 in USD        ", round(ex.rate({"X1": +1, "USD": -1}), 4))
    print()
 
    # A one-sided move: the buyer walks his quote up, the seller stays put.
    for step_no in range(1, 4):
        ex.submit(Order(weights={"X1": +1, "RUB": -1}, z=110.0, lam=1e6,
                        agent="buyer", max_notional=5e6))
        rep = ex.step(dt=1.0)
        print(f"t={ex.t:>4.0f}  X1={rep.prices['X1']:8.4f}  "
              f"X2={rep.prices['X2']:8.4f}  "
              f"imbalance/gross={rep.max_value_imbalance / rep.gross_notional:.1e}")
    print()
 
    for agent in ("buyer", "seller", "arb"):
        bal = {a: round(q, 3) for a, q in ex.balances[agent].items()}
        print(f"  {agent:<7} mtm={ex.mark_to_market(agent):+14.2f} RUB   {bal}")