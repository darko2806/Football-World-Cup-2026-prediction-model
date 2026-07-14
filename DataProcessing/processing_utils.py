import pandas as pd
import numpy as np

def add_strength_adjusted_margin_ema(data, span = 33, rank_scale = 50.0, fill_initial = True):
    
    data = data.copy()

    data['date'] = pd.to_datetime(data['date'])
    data = data.sort_values('date').reset_index(drop=True)

    alpha = 2 / (span + 1)

    team_ema = {}

    home_ema_values = []
    away_ema_values = []

    for _, row in data.iterrows():
        home_team = row['home_team']
        away_team = row['away_team']

         # Historical values before the current match
        home_ema_values.append(team_ema.get(home_team, np.nan))
        away_ema_values.append(team_ema.get(away_team, np.nan))

        actual_margin_home = row["home_score"] - row["away_score"]

        # rank_advantage = away_rank - home_rank
        # Positive value means home team is stronger
        expected_margin_home = row["rank_advantage"] / rank_scale

        strength_adjusted_margin_home = (
            actual_margin_home - expected_margin_home
        )

        strength_adjusted_margin_away = (
            -strength_adjusted_margin_home
        )

        old_home_ema = team_ema.get(home_team, np.nan)
        old_away_ema = team_ema.get(away_team, np.nan)

        if pd.isna(old_home_ema):
            team_ema[home_team] = strength_adjusted_margin_home
        else:
            team_ema[home_team] = (
                alpha * strength_adjusted_margin_home
                + (1 - alpha) * old_home_ema
            )

        if pd.isna(old_away_ema):
            team_ema[away_team] = strength_adjusted_margin_away
        else:
            team_ema[away_team] = (
                alpha * strength_adjusted_margin_away
                + (1 - alpha) * old_away_ema
            )

    home_col = f"home_strength_adjusted_margin_ema{span}"
    away_col = f"away_strength_adjusted_margin_ema{span}"
    diff_col = f"strength_adjusted_margin_diff_ema{span}"

    data[home_col] = home_ema_values
    data[away_col] = away_ema_values

    data[diff_col] = data[home_col] - data[away_col]

    if fill_initial:
        data[[home_col, away_col, diff_col]] = data[
            [home_col, away_col, diff_col]
        ].fillna(0.0)

    return data       
