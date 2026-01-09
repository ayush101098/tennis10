"""
Dashboard Components - Reusable UI elements
===========================================
"""

from dashboard.components.match_card import (
    render_match_card,
    render_detailed_match_view,
    render_compact_match_row
)

from dashboard.components.charts import (
    create_pnl_chart,
    create_drawdown_chart,
    create_model_comparison_radar,
    create_calibration_plot,
    create_edge_distribution_chart,
    create_odds_movement_chart,
    create_roi_by_confidence_chart,
    create_win_rate_by_surface_chart
)

from dashboard.components.tables import (
    render_predictions_table,
    render_bets_table,
    render_performance_summary,
    render_model_performance_table,
    render_player_stats_table,
    render_upcoming_matches_table
)

__all__ = [
    # Match cards
    'render_match_card',
    'render_detailed_match_view',
    'render_compact_match_row',
    
    # Charts
    'create_pnl_chart',
    'create_drawdown_chart',
    'create_model_comparison_radar',
    'create_calibration_plot',
    'create_edge_distribution_chart',
    'create_odds_movement_chart',
    'create_roi_by_confidence_chart',
    'create_win_rate_by_surface_chart',
    
    # Tables
    'render_predictions_table',
    'render_bets_table',
    'render_performance_summary',
    'render_model_performance_table',
    'render_player_stats_table',
    'render_upcoming_matches_table'
]
