"""Live market engine — provider-independent ingestion layer.

Scope note: this package is the TRANSPORT and INTEGRITY layer only. Pricing
already exists in this repo and is not reimplemented here —
`execution/inplay.py` (score-aware Markov true_p), `execution/momentum.py`
(live momentum), `execution/edgescore.py` (uncertainty gate) and
`execution/signals_gen.py` (signal emission) are the model. Anything in this
package that starts computing a probability is a bug.
"""

from execution.live.events import (  # noqa: F401
    EventType, LiveEvent, Score, P1, P2, derive_transitions,
)
from execution.live.feed import (  # noqa: F401
    FeedStatus, Health, LatencyBreakdown, SequenceTracker, health_for_age, worst,
)
from execution.live.provider import (  # noqa: F401
    ReplayProvider, ScriptedEvent, TennisDataProvider,
)
from execution.live.odds import (  # noqa: F401
    BookmakerQuote, ExchangeQuote, MarketView,
)
from execution.live.state import (  # noqa: F401
    InMemoryStateStore, MatchState, MatchStateMachine, RedisStateStore, StateStore,
)
from execution.live.engine import Fair, GameLadder, ModelBridge, game_ladder  # noqa: F401
from execution.live.signals import Signal, SignalEngine, SignalStatus  # noqa: F401
from execution.live.gateway import RoomRegistry, Viewer, build_payload  # noqa: F401
from execution.live.runtime import LiveRuntime, MatchContext  # noqa: F401
