"""TrainingLog — self-hosted RL experiment tracker."""

__version__ = "0.1.0"

from .sdk.importer import import_league_db
from .sdk.league import (
    log_checkpoint,
    log_match,
    log_match_series,
    log_plasticity,
    log_rating,
)
from .sdk.meta import import_meta_games_ndjson, log_card_catalog, log_meta_games
from .sdk.replays import log_card_stats, log_distribution, log_replay
from .sdk.run import Command, Run, SummaryWriter, init, sync

__all__ = [
    "__version__",
    "Command",
    "Run",
    "SummaryWriter",
    "import_league_db",
    "import_meta_games_ndjson",
    "init",
    "log_card_catalog",
    "log_card_stats",
    "log_checkpoint",
    "log_distribution",
    "log_match",
    "log_match_series",
    "log_meta_games",
    "log_plasticity",
    "log_rating",
    "log_replay",
    "sync",
]
