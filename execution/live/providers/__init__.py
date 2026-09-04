"""Provider adapters. One module per vendor; none of them are imported by the
model, which only ever sees `LiveEvent` (see execution/live/provider.py)."""

from execution.live.providers.failover import (  # noqa: F401
    FailoverManager, ProviderHealth, ProviderStats,
)
from execution.live.providers.livesport import LivesportProvider  # noqa: F401
