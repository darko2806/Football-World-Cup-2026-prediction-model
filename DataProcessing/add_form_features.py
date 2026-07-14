"""
Adds team "form" features (SMA/EMA over match points W=1/D=0/L=-1) to the
fifa_and_elo_rankings_clean_filled dataset, used only for Eksperiment 2
(forma-only models).

Forma se racuna po timu iz SVIH njegovih prethodnih utakmica (domacih i
gostujucih pomesano, hronoloski), koristeci iskljucivo mecheve odigrane
PRE trenutnog (shift(1)) - bez data leakage-a. Poeni po utakmici su
W=1 / D=0 / L=-1 (isto bodovanje kao kolona "result"), a ne standardno
3/1/0, da bi niz ostao simetrican bez dodatnog rasprsivanja varijanse.
"""

from pathlib import Path

import pandas as pd

SOURCE_PATH = (
    Path(__file__).resolve().parent.parent
    / "data" / "Processed_Project_Data" / "fifa_and_elo_rankings_clean_filled.csv"
)
OUTPUT_PATH = (
    Path(__file__).resolve().parent.parent
    / "data" / "Processed_Project_Data" / "fifa_and_elo_with_ema.csv"
)

SMA_WINDOWS = [3, 5, 7]
EMA_ALPHAS = {"ema50": 0.5, "ema33": 1 / 3, "ema25": 0.25}



df = pd.read_csv(SOURCE_PATH)
df = df.drop(columns=["Unnamed: 0"], errors="ignore")
df["date"] = pd.to_datetime(df["date"], format="mixed")



home_long = df[["match_id", "date", "home_team", "result"]].rename(columns={"home_team": "team"})
home_long["points"] = home_long["result"]

away_long = df[["match_id", "date", "away_team", "result"]].rename(columns={"away_team": "team"})
away_long["points"] = -away_long["result"]

team_long = pd.concat([home_long, away_long], ignore_index=True)
team_long = team_long.sort_values(["team", "date", "match_id"]).reset_index(drop=True)


grouped = team_long.groupby("team")["points"]

for window in SMA_WINDOWS:
    team_long[f"form_sma{window}"] = grouped.transform(
        lambda s, w=window: s.shift(1).rolling(window=w, min_periods=1).mean()
    )

for name, alpha in EMA_ALPHAS.items():
    team_long[f"form_{name}"] = grouped.transform(
        lambda s, a=alpha: s.shift(1).ewm(alpha=a, adjust=False).mean()
    )

FORM_COLS = [f"form_sma{w}" for w in SMA_WINDOWS] + [f"form_{n}" for n in EMA_ALPHAS]


home_form = (
    team_long.rename(columns={"team": "home_team"})[["match_id", "home_team"] + FORM_COLS]
    .rename(columns={c: f"home_{c}" for c in FORM_COLS})
)
away_form = (
    team_long.rename(columns={"team": "away_team"})[["match_id", "away_team"] + FORM_COLS]
    .rename(columns={c: f"away_{c}" for c in FORM_COLS})
)

df = df.merge(home_form, on=["match_id", "home_team"], how="left")
df = df.merge(away_form, on=["match_id", "away_team"], how="left")


for c in FORM_COLS:
    suffix = c.replace("form_", "")
    df[f"form_diff_{suffix}"] = df[f"home_{c}"] - df[f"away_{c}"]


df.to_csv(OUTPUT_PATH)

print(f"Dodato {2 * len(FORM_COLS) + len(FORM_COLS)} novih kolona forme.")
print(f"Ukupno kolona: {len(df.columns)}, redova: {len(df)}")
print("Saved:", OUTPUT_PATH)
