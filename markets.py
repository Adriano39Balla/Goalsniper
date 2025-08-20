# app/markets.py

"""
Market definitions for Goalsniper training.
Each market defines:
- label: column name used in DataFrame
- label_fn: how to compute label from final result (gh, ga)
"""

MARKETS = {
    "O25": {
        "label": "label_o25",
        "label_fn": lambda gh, ga: int((gh + ga) >= 3),
    },
    "BTTS_YES": {
        "label": "label_btts",
        "label_fn": lambda gh, ga: int(gh > 0 and ga > 0),
    },
    # 🔮 Ready for extension:
    # "HOME_WIN": {"label": "label_home_win", "label_fn": lambda gh, ga: int(gh > ga)},
    # "AWAY_WIN": {"label": "label_away_win", "label_fn": lambda gh, ga: int(ga > gh)},
    # "DRAW": {"label": "label_draw", "label_fn": lambda gh, ga: int(gh == ga)},
    # "HANDICAP_H-1.5": {"label": "label_handicap_h-1.5", "label_fn": lambda gh, ga: int((gh - 1.5) > ga)},
}
