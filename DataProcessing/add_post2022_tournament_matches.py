"""
Adds matches from major continental tournaments played after 2022
(UEFA Euro, African Cup of Nations, AFC Asian Cup, Copa America)
for teams that qualified for the 2026 FIFA World Cup, to the
fifa_and_elo_rankings_clean_filled dataset.

Follows the exact same processing steps as DataProcessingForModel.ipynb:
1. Add FIFA ranking features (time-aware merge_asof)
2. Add ELO features (current snapshots + historical fallback fill)
3. Compute rank_advantage, points_diff, elo_advantage, result
4. Drop rows with missing rankings / bad ELO dates / non-positive ELO
5. Append to the existing clean_filled dataset
"""

import pandas as pd


WC2026_TEAMS = [
    'Algeria', 'Argentina', 'Australia', 'Austria', 'Belgium', 'Bosnia and Herzegovina',
    'Brazil', 'Canada', 'Cape Verde', 'Colombia', 'Croatia', 'Curaçao', 'Czech Republic',
    'DR Congo', 'Ecuador', 'Egypt', 'England', 'France', 'Germany', 'Ghana', 'Haiti',
    'Iran', 'Iraq', 'Ivory Coast', 'Japan', 'Jordan', 'Mexico', 'Morocco', 'Netherlands',
    'New Zealand', 'Norway', 'Panama', 'Paraguay', 'Portugal', 'Qatar', 'Saudi Arabia',
    'Scotland', 'Senegal', 'South Africa', 'South Korea', 'Spain', 'Sweden', 'Switzerland',
    'Tunisia', 'Turkey', 'United States', 'Uruguay', 'Uzbekistan'
]

TOURNAMENTS = ['UEFA Euro', 'African Cup of Nations', 'AFC Asian Cup', 'Copa América']


final_existing = pd.read_csv('../data/Processed_Project_Data/fifa_and_elo_rankings_clean_filled.csv')
final_existing = final_existing.drop(columns=["Unnamed: 0"], errors="ignore")
max_match_id = final_existing["match_id"].max()


results = pd.read_csv('../data/raw_data/results.csv')
results["date"] = pd.to_datetime(results["date"])

new_matches = results[
    (results["date"] > "2022-12-31") &
    (results["tournament"].isin(TOURNAMENTS)) &
    (results["home_team"].isin(WC2026_TEAMS) | results["away_team"].isin(WC2026_TEAMS))
].copy()

new_matches["neutral"] = new_matches["neutral"].map({True: 1, False: 0, "TRUE": 1, "FALSE": 0})
new_matches["neutral"] = new_matches["neutral"].fillna(new_matches["neutral"]).astype(int)
new_matches["year"] = new_matches["date"].dt.year
new_matches["is_world_cup"] = 0
new_matches["is_qualifier"] = 0

new_matches = new_matches.sort_values(["date", "home_team"]).reset_index(drop=True)
new_matches["match_id"] = range(max_match_id + 1, max_match_id + 1 + len(new_matches))

print(f"New matches selected: {len(new_matches)}")
print(new_matches["tournament"].value_counts())


rank_data = pd.read_csv('../data/ProcessedDataForTestModel/FIFA_rankings_2021_2026.csv')
rank_data = rank_data.drop(columns=["Unnamed: 0"], errors="ignore")
rank_data["rank_date"] = pd.to_datetime(rank_data["rank_date"])

TEAM_NAME_FIXES = {
    "Brunei": "Brunei Darussalam",
    "Cape Verde": "Cabo Verde",
    "DR Congo": "Congo DR",
    "Iran": "IR Iran",
    "Ivory Coast": "Côte d'Ivoire",
    "Kyrgyzstan": "Kyrgyz Republic",
    "North Korea": "Korea DPR",
    "South Korea": "Korea Republic",
    "Taiwan": "Chinese Taipei",
    "United States": "USA",
    "United States Virgin Islands": "US Virgin Islands",
    "Saint Kitts and Nevis": "St Kitts and Nevis",
    "Saint Lucia": "St Lucia",
    "Saint Vincent and the Grenadines": "St Vincent and the Grenadines",
}

new_matches["home_team_rank_name"] = new_matches["home_team"].replace(TEAM_NAME_FIXES)
new_matches["away_team_rank_name"] = new_matches["away_team"].replace(TEAM_NAME_FIXES)

ranking_data = rank_data[
    [
        "rank_date",
        "country",
        "country_code",
        "rank",
        "previous_rank",
        "rank_change",
        "total_points",
        "previous_points"
    ]
].copy()

ranking_data = ranking_data.rename(columns={"country": "team_rank_name"})


def add_ranking_to_matches(matches_df, team_col, prefix):
    left = matches_df[["match_id", "date", team_col]].copy()
    left = left.rename(columns={team_col: "team_rank_name"})

    right = ranking_data.copy()

    left = left.sort_values(["date", "team_rank_name"])
    right = right.sort_values(["rank_date", "team_rank_name"])

    merged = pd.merge_asof(
        left, right,
        left_on="date", right_on="rank_date",
        by="team_rank_name", direction="backward"
    )

    merged = merged.rename(columns={
        "rank_date": f"{prefix}_rank_date_used",
        "country_code": f"{prefix}_country_code",
        "rank": f"{prefix}_rank",
        "previous_rank": f"{prefix}_previous_rank",
        "rank_change": f"{prefix}_rank_change",
        "total_points": f"{prefix}_total_points",
        "previous_points": f"{prefix}_previous_points"
    })

    return merged[[
        "match_id",
        f"{prefix}_rank_date_used", f"{prefix}_country_code", f"{prefix}_rank",
        f"{prefix}_previous_rank", f"{prefix}_rank_change",
        f"{prefix}_total_points", f"{prefix}_previous_points"
    ]]


home_ranking = add_ranking_to_matches(new_matches, "home_team_rank_name", "home")
away_ranking = add_ranking_to_matches(new_matches, "away_team_rank_name", "away")

new_matches = new_matches.merge(home_ranking, on="match_id", how="left")
new_matches = new_matches.merge(away_ranking, on="match_id", how="left")

new_matches["rank_advantage"] = new_matches["away_rank"] - new_matches["home_rank"]
new_matches["points_diff"] = new_matches["home_total_points"] - new_matches["away_total_points"]


def get_match_result(row):
    if row["home_score"] > row["away_score"]:
        return 1
    elif row["home_score"] < row["away_score"]:
        return -1
    else:
        return 0


new_matches["result"] = new_matches.apply(get_match_result, axis=1)

missing_rankings = new_matches[
    new_matches["home_rank"].isna() | new_matches["away_rank"].isna()
][["date", "home_team", "away_team", "home_rank", "away_rank"]]
print("\nMatches with missing FIFA ranking:", len(missing_rankings))
print(missing_rankings)

new_matches = new_matches.drop(
    columns=["home_team_rank_name", "away_team_rank_name", "home_country_code", "away_country_code"],
    errors="ignore"
)


elo_data = pd.read_csv('../data/raw_data/elo_ratings_wc2026.csv')
elo_data["snapshot_date"] = pd.to_datetime(elo_data["snapshot_date"])

elo_data = elo_data[[
    "snapshot_date", "country", "rank", "rating", "rank_avg", "rating_avg",
    "matches_total", "wins", "losses", "draws", "goals_for", "goals_against",
    "confederation", "is_host"
]].copy()

elo_data = elo_data.rename(columns={
    "snapshot_date": "elo_date",
    "country": "team_elo_name",
    "rank": "elo_rank",
    "rating": "elo_rating",
    "rank_avg": "elo_avg_rank",
    "rating_avg": "elo_avg_rating"
})

elo_data = elo_data[elo_data["elo_date"] >= "2021-01-01"].copy()

TEAM_NAME_FIXES_ELO = {
    "USA": "United States",
    "United States": "United States",

    "IR Iran": "Iran",
    "Iran": "Iran",

    "Korea Republic": "South Korea",
    "South Korea": "South Korea",

    "Korea DPR": "North Korea",
    "North Korea": "North Korea",

    "Congo DR": "DR Congo",
    "DR Congo": "DR Congo",

    "Côte d'Ivoire": "Ivory Coast",
    "Ivory Coast": "Ivory Coast",

    "Cabo Verde": "Cape Verde",
    "Cape Verde": "Cape Verde",

    "Kyrgyz Republic": "Kyrgyzstan",
    "Kyrgyzstan": "Kyrgyzstan",
}

new_matches["home_team_elo_name"] = new_matches["home_team"].replace(TEAM_NAME_FIXES_ELO)
new_matches["away_team_elo_name"] = new_matches["away_team"].replace(TEAM_NAME_FIXES_ELO)


def add_elo_to_matches(matches_df, team_col, prefix):
    left = matches_df[["match_id", "date", team_col]].copy()
    left = left.rename(columns={team_col: "team_elo_name"})

    right = elo_data.copy()

    left = left.sort_values(["date", "team_elo_name"])
    right = right.sort_values(["elo_date", "team_elo_name"])

    merged = pd.merge_asof(
        left, right,
        left_on="date", right_on="elo_date",
        by="team_elo_name", direction="backward"
    )

    merged = merged.rename(columns={
        "elo_date": f"{prefix}_elo_date_used",
        "elo_rank": f"{prefix}_elo_rank",
        "elo_rating": f"{prefix}_elo",
        "elo_avg_rank": f"{prefix}_elo_avg_rank",
        "elo_avg_rating": f"{prefix}_elo_avg_rating",
        "matches_total": f"{prefix}_elo_matches_total",
        "wins": f"{prefix}_elo_wins",
        "losses": f"{prefix}_elo_losses",
        "draws": f"{prefix}_elo_draws",
        "goals_for": f"{prefix}_elo_goals_for",
        "goals_against": f"{prefix}_elo_goals_against",
        "confederation": f"{prefix}_confederation",
        "is_host": f"{prefix}_is_host"
    })

    return merged[[
        "match_id",
        f"{prefix}_elo_date_used", f"{prefix}_elo_rank", f"{prefix}_elo",
        f"{prefix}_elo_avg_rank", f"{prefix}_elo_avg_rating", f"{prefix}_elo_matches_total",
        f"{prefix}_elo_wins", f"{prefix}_elo_losses", f"{prefix}_elo_draws",
        f"{prefix}_elo_goals_for", f"{prefix}_elo_goals_against",
        f"{prefix}_confederation", f"{prefix}_is_host"
    ]]


home_elo = add_elo_to_matches(new_matches, "home_team_elo_name", "home")
away_elo = add_elo_to_matches(new_matches, "away_team_elo_name", "away")

new_matches = new_matches.merge(home_elo, on="match_id", how="left")
new_matches = new_matches.merge(away_elo, on="match_id", how="left")

new_matches["elo_advantage"] = new_matches["home_elo"] - new_matches["away_elo"]
new_matches["elo_rank_advantage"] = new_matches["away_elo_rank"] - new_matches["home_elo_rank"]
new_matches["elo_avg_rating_advantage"] = new_matches["home_elo_avg_rating"] - new_matches["away_elo_avg_rating"]
new_matches["elo_win_rate_home"] = new_matches["home_elo_wins"] / new_matches["home_elo_matches_total"]
new_matches["elo_win_rate_away"] = new_matches["away_elo_wins"] / new_matches["away_elo_matches_total"]
new_matches["elo_win_rate_advantage"] = new_matches["elo_win_rate_home"] - new_matches["elo_win_rate_away"]

missing_elo = new_matches[
    new_matches["home_elo"].isna() | new_matches["away_elo"].isna()
][["date", "home_team", "away_team", "home_elo", "away_elo"]]
print("\nMatches with missing ELO (before historical fallback):", len(missing_elo))

new_matches = new_matches.drop(
    columns=["home_team_elo_name", "away_team_elo_name", "home_confederation", "away_confederation"],
    errors="ignore"
)


import re

elo_hist = pd.read_csv('../data/raw_data/eloratings.csv')
elo_hist["date"] = pd.to_datetime(elo_hist["date"], format="mixed", errors="coerce")


def clean_team_name(name):
    if pd.isna(name):
        return name
    name = str(name)
    name = name.replace("\xa0", " ")
    name = re.sub(r"\s+", " ", name)
    return name.strip()


new_matches["home_team_elo_name_fill"] = new_matches["home_team"].apply(clean_team_name)
new_matches["away_team_elo_name_fill"] = new_matches["away_team"].apply(clean_team_name)
elo_hist["team_elo_name"] = elo_hist["team"].apply(clean_team_name)

TEAM_NAME_FIXES_ELO_HIST = {
    "American Samoa": "Eastern Samoa",
    "China PR": "China",
    "Czech Republic": "Czechia",
    "DR Congo": "Democratic Republic of Congo",
    "Macau": "Macao",
    "Republic of Ireland": "Ireland",
    "São Tomé and Príncipe": "Sao Tome and Principe",
    "Timor-Leste": "East Timor",
    "United States Virgin Islands": "US Virgin Islands",

    "USA": "United States",
    "Korea Republic": "South Korea",
    "Korea DPR": "North Korea",
    "IR Iran": "Iran",
    "Congo DR": "Democratic Republic of Congo",
    "Côte d'Ivoire": "Ivory Coast",
    "Cabo Verde": "Cape Verde",
    "Kyrgyz Republic": "Kyrgyzstan",
    "UAE": "United Arab Emirates",
}

new_matches["home_team_elo_name_fill"] = new_matches["home_team_elo_name_fill"].replace(TEAM_NAME_FIXES_ELO_HIST)
new_matches["away_team_elo_name_fill"] = new_matches["away_team_elo_name_fill"].replace(TEAM_NAME_FIXES_ELO_HIST)

elo_hist = elo_hist[["date", "team_elo_name", "rating", "change"]].copy()
elo_hist = elo_hist.rename(columns={"date": "elo_date", "rating": "elo_rating", "change": "elo_change"})
elo_hist = elo_hist.dropna(subset=["elo_date", "elo_rating"])
elo_hist = elo_hist.sort_values(["elo_date", "team_elo_name"]).drop_duplicates(
    ["team_elo_name", "elo_date"], keep="last"
)


def get_historical_elo(matches_df, team_col, prefix):
    left = matches_df[["match_id", "date", team_col]].copy()
    left = left.rename(columns={team_col: "team_elo_name"})

    left = left.sort_values(["date", "team_elo_name"]).reset_index(drop=True)
    right = elo_hist.sort_values(["elo_date", "team_elo_name"]).reset_index(drop=True)

    previous_elo = pd.merge_asof(
        left, right,
        left_on="date", right_on="elo_date",
        by="team_elo_name", direction="backward",
        allow_exact_matches=False
    )
    previous_elo["elo_source_fill"] = "historical_previous"

    future_elo = pd.merge_asof(
        left, right,
        left_on="date", right_on="elo_date",
        by="team_elo_name", direction="forward",
        allow_exact_matches=True
    )

    missing_mask = previous_elo["elo_rating"].isna()

    previous_elo.loc[missing_mask, "elo_date"] = future_elo.loc[missing_mask, "elo_date"]
    previous_elo.loc[missing_mask, "elo_rating"] = future_elo.loc[missing_mask, "elo_rating"]
    previous_elo.loc[missing_mask, "elo_change"] = future_elo.loc[missing_mask, "elo_change"]

    previous_elo.loc[
        missing_mask & future_elo["elo_rating"].notna(), "elo_source_fill"
    ] = "historical_future_fallback"

    previous_elo = previous_elo.rename(columns={
        "elo_date": f"{prefix}_elo_date_fill",
        "elo_rating": f"{prefix}_elo_fill",
        "elo_change": f"{prefix}_elo_change_fill",
        "elo_source_fill": f"{prefix}_elo_source_fill"
    })

    return previous_elo[[
        "match_id", f"{prefix}_elo_date_fill", f"{prefix}_elo_fill",
        f"{prefix}_elo_change_fill", f"{prefix}_elo_source_fill"
    ]]


home_fill = get_historical_elo(new_matches, "home_team_elo_name_fill", "home")
away_fill = get_historical_elo(new_matches, "away_team_elo_name_fill", "away")

new_matches = new_matches.merge(home_fill, on="match_id", how="left")
new_matches = new_matches.merge(away_fill, on="match_id", how="left")

new_matches["home_elo"] = new_matches["home_elo"].fillna(new_matches["home_elo_fill"])
new_matches["away_elo"] = new_matches["away_elo"].fillna(new_matches["away_elo_fill"])
new_matches["home_elo_date_used"] = new_matches["home_elo_date_used"].fillna(new_matches["home_elo_date_fill"])
new_matches["away_elo_date_used"] = new_matches["away_elo_date_used"].fillna(new_matches["away_elo_date_fill"])

new_matches["elo_advantage"] = new_matches["home_elo"] - new_matches["away_elo"]

print("\nHome ELO still missing:", new_matches["home_elo"].isna().sum())
print("Away ELO still missing:", new_matches["away_elo"].isna().sum())

temp_cols = [
    "home_team_elo_name_fill", "away_team_elo_name_fill",
    "home_elo_date_fill", "home_elo_fill", "home_elo_change_fill", "home_elo_source_fill",
    "away_elo_date_fill", "away_elo_fill", "away_elo_change_fill", "away_elo_source_fill",
]
new_matches = new_matches.drop(columns=temp_cols, errors="ignore")


new_matches["date"] = pd.to_datetime(new_matches["date"])
new_matches["home_elo_date_used"] = pd.to_datetime(new_matches["home_elo_date_used"])
new_matches["away_elo_date_used"] = pd.to_datetime(new_matches["away_elo_date_used"])

before = len(new_matches)

new_matches_clean = new_matches[
    ~(
        (new_matches["home_elo_date_used"] > new_matches["date"]) |
        (new_matches["away_elo_date_used"] > new_matches["date"]) |
        new_matches["home_rank"].isna() |
        new_matches["away_rank"].isna() |
        (new_matches["home_elo"] <= 0) |
        (new_matches["away_elo"] <= 0) |
        new_matches["home_elo"].isna() |
        new_matches["away_elo"].isna()
    )
].copy()

new_matches_clean["elo_advantage"] = new_matches_clean["home_elo"] - new_matches_clean["away_elo"]

print(f"\nRows before cleaning: {before}")
print(f"Rows after cleaning: {len(new_matches_clean)}")


FINAL_COLUMNS = [
    'date', 'home_team', 'away_team', 'home_score', 'away_score',
    'tournament', 'city', 'country', 'neutral', 'year', 'is_world_cup',
    'is_qualifier', 'match_id', 'home_rank_date_used', 'home_rank',
    'home_previous_rank', 'home_rank_change', 'home_total_points',
    'home_previous_points', 'away_rank_date_used', 'away_rank',
    'away_previous_rank', 'away_rank_change', 'away_total_points',
    'away_previous_points', 'rank_advantage', 'points_diff',
    'home_elo', 'away_elo', 'elo_advantage', 'result'
]

new_matches_final = new_matches_clean[FINAL_COLUMNS].copy()
new_matches_final["match_id"] = range(max_match_id + 1, max_match_id + 1 + len(new_matches_final))

combined = pd.concat([final_existing[FINAL_COLUMNS], new_matches_final], ignore_index=True)

combined.to_csv('../data/Processed_Project_Data/fifa_and_elo_rankings_clean_filled.csv')

print(f"\nFinal dataset rows: {len(combined)} (was {len(final_existing)}, added {len(new_matches_final)})")
print("Saved: ../data/Processed_Project_Data/fifa_and_elo_rankings_clean_filled.csv")
