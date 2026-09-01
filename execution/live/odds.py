"""Market prices (PRD §9, §10, §11) — bookmakers and exchanges, kept apart.

WHY THE SEPARATION IS STRUCTURAL AND NOT A STYLE CHOICE
    A bookmaker quotes a price it wants you to take, with margin baked in. An
    exchange shows what other people are willing to trade, with a spread and a
    depth behind it. They are different measurements of different things:

        BOOKMAKER   decimal odds -> implied probability -> de-vig -> fair
        EXCHANGE    bid / ask    -> mid                 -> spread, depth

    Averaging them produces a number that is neither. Worse, an exchange mid
    taken from a one-tick market with $12 behind it looks identical to a deep
    one once it is a float, and only the depth tells you the price is not real.
    So they are separate types, and combining them is an explicit act with a
    documented rule rather than an accident of both being `float`.

RELEVANCE HERE
    This product's only working price is currently an exchange (Polymarket);
    the SofaScore bookmaker feed answers 403 and its bulk feed returns a
    disjoint id space. The bookmaker path is therefore written and tested but
    largely unfed today — which is exactly why it must not be silently
    conflated with the exchange path when it does come back.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from execution.live.feed import Health, health_for_age

# A two-way book whose de-vigged probabilities do not roughly sum to 1 is not a
# two-way book — most likely the two legs came from different markets.
DEVIG_TOLERANCE = 0.12

# Exchange spreads wider than this mean the mid is a fiction: there is no
# price, only two hopeful orders far apart.
MAX_TRADEABLE_SPREAD = 0.10

# Minimum resting size (in quote currency) for a side to count as real.
MIN_DEPTH = 50.0


@dataclass(frozen=True)
class BookmakerQuote:
    """One book's two-way price on one market."""

    book: str
    market: str                       # "match_winner" | "set_winner" | "game_winner"
    odds_p1: float
    odds_p2: float
    ts_ms: int = field(default_factory=lambda: int(time.time() * 1000))

    @property
    def valid(self) -> bool:
        # Decimal odds of 1.0 or less are not a price; they are a parse error.
        return self.odds_p1 > 1.0 and self.odds_p2 > 1.0

    def devigged(self) -> Optional[tuple[float, float]]:
        """Margin-free implied probabilities (§11).

        Proportional (multiplicative) de-vig: divide each raw implied
        probability by the overround. It assumes margin is applied evenly
        across both legs, which is not exactly true — books shade the favourite
        less — but the alternatives need parameters we cannot fit from a single
        two-way price, and a wrong parameter is worse than a known-simple rule.
        """
        if not self.valid:
            return None
        raw1, raw2 = 1.0 / self.odds_p1, 1.0 / self.odds_p2
        total = raw1 + raw2
        # Overround below 1 means we are being offered free money by both
        # sides at once, which never happens — it means bad data.
        if total <= 1.0 - DEVIG_TOLERANCE or total > 1.0 + 1.0:
            return None
        return raw1 / total, raw2 / total

    @property
    def overround(self) -> Optional[float]:
        if not self.valid:
            return None
        return 1.0 / self.odds_p1 + 1.0 / self.odds_p2 - 1.0


@dataclass(frozen=True)
class ExchangeQuote:
    """One exchange's book on one side of one market.

    Prices are probabilities (0-1), the convention Polymarket and Betfair's
    implied form both reduce to. `bid` is the best price you can sell at, `ask`
    the best you can buy at — so `ask > bid` always, and the crossed case is
    reported as untradeable rather than silently averaged.
    """

    venue: str
    market: str
    side: str                          # "p1" | "p2"
    bid: Optional[float] = None
    ask: Optional[float] = None
    bid_size: float = 0.0
    ask_size: float = 0.0
    ts_ms: int = field(default_factory=lambda: int(time.time() * 1000))

    @property
    def spread(self) -> Optional[float]:
        if self.bid is None or self.ask is None:
            return None
        return self.ask - self.bid

    @property
    def mid(self) -> Optional[float]:
        """Midpoint, or the single side we have if only one is quoted.

        A one-sided book is a real state — nobody is offering the other side —
        and returning that price with `tradeable=False` is more honest than
        returning None and losing the information.
        """
        if self.bid is not None and self.ask is not None:
            if self.ask < self.bid:
                return None                     # crossed book: not a price
            return (self.bid + self.ask) / 2.0
        return self.bid if self.bid is not None else self.ask

    @property
    def tradeable(self) -> bool:
        """Whether this quote is something you could actually transact against.

        A mid with no depth behind it is the most dangerous number in this
        module: it looks exactly like a real price once it is a float, and an
        edge computed against it is an edge against nobody.
        """
        s = self.spread
        if s is None or s < 0 or s > MAX_TRADEABLE_SPREAD:
            return False
        return self.bid_size >= MIN_DEPTH and self.ask_size >= MIN_DEPTH


@dataclass
class MarketView:
    """Everything we know about one market on one match, from all sources.

    Bookmaker consensus and exchange consensus are computed separately and
    exposed separately. `fair()` is the one place they are allowed to meet,
    and it states its preference rather than averaging.
    """

    match_id: str
    market: str
    books: list = field(default_factory=list)       # list[BookmakerQuote]
    exchange: dict = field(default_factory=dict)    # side -> ExchangeQuote
    updated_ms: int = 0

    def add_book(self, q: BookmakerQuote) -> None:
        self.books = [b for b in self.books if b.book != q.book] + [q]
        self.updated_ms = max(self.updated_ms, q.ts_ms)

    def add_exchange(self, q: ExchangeQuote) -> None:
        self.exchange[q.side] = q
        self.updated_ms = max(self.updated_ms, q.ts_ms)

    def book_consensus(self) -> Optional[tuple[float, float]]:
        """Median de-vigged probability across books.

        Median, not mean: one book with a stale or fat-fingered line should not
        drag the consensus, and with a handful of books the median is the
        cheapest robust estimator available.
        """
        pairs = [q.devigged() for q in self.books]
        p1s = sorted(p[0] for p in pairs if p)
        if not p1s:
            return None
        mid = p1s[len(p1s) // 2] if len(p1s) % 2 else (p1s[len(p1s) // 2 - 1] + p1s[len(p1s) // 2]) / 2
        return mid, 1.0 - mid

    def exchange_consensus(self) -> Optional[tuple[float, float]]:
        """De-vigged two-sided exchange price.

        Both sides of an exchange sum to slightly more than 1 for the same
        reason a book's do — the spread — so the pair is normalised the same
        way. With only one side quoted we infer the other as its complement,
        which is exact for a two-outcome market.
        """
        q1, q2 = self.exchange.get("p1"), self.exchange.get("p2")
        m1 = q1.mid if q1 else None
        m2 = q2.mid if q2 else None
        if m1 is None and m2 is None:
            return None
        if m1 is None:
            m1 = 1.0 - m2                       # type: ignore[operator]
        if m2 is None:
            m2 = 1.0 - m1
        total = m1 + m2
        if total <= 0:
            return None
        return m1 / total, m2 / total

    def fair(self) -> Optional[tuple[float, float, str]]:
        """The market probability to price against, and where it came from.

        Preference order, and the reason for it: an exchange price with real
        depth is money other people have actually risked, which is a stronger
        statement than a bookmaker's posted line. A bookmaker consensus across
        several books is next. A thin exchange quote is used only when there is
        nothing else, and is labelled so the caller can widen its uncertainty.
        """
        ex = self.exchange_consensus()
        ex_tradeable = all(q.tradeable for q in self.exchange.values()) and bool(self.exchange)
        if ex and ex_tradeable:
            return ex[0], ex[1], "exchange"
        bc = self.book_consensus()
        if bc:
            return bc[0], bc[1], "bookmaker"
        if ex:
            return ex[0], ex[1], "exchange_thin"
        return None

    @property
    def has_price(self) -> bool:
        """Whether any source has ever quoted this market.

        NOT the same as being fresh. A market nobody has quoted is absent; a
        market last quoted four minutes ago is stale. Collapsing the two makes
        a perfectly healthy scoreboard report OFFLINE simply because no odds
        feed is attached, which is how a working system gets diagnosed as
        broken.
        """
        return self.updated_ms > 0

    def health(self, now_ms: Optional[int] = None) -> Health:
        """Freshness of the PRICE. Only meaningful when `has_price`."""
        now = now_ms if now_ms is not None else int(time.time() * 1000)
        if not self.updated_ms:
            return Health.OFFLINE
        return health_for_age(now - self.updated_ms)


def implied(odds: float) -> Optional[float]:
    """Decimal odds to raw implied probability. None for a non-price."""
    return 1.0 / odds if odds and odds > 1.0 else None


def to_decimal(p: float) -> Optional[float]:
    """Probability to decimal odds — the inverse, for display."""
    return 1.0 / p if 0.0 < p < 1.0 else None
