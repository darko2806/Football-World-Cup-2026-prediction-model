# SP 2026 Predikcioni Model

## Arhitektura projekta
- Model: XGBoost regresija
- Pristup: Predviđamo lambda_home i lambda_away (očekivani golovi)
- Iz lambdi → Poisson matrica rezultata → P(W), P(D), P(L)
- Target: normalizovane prosečne kvote više kladionica (skinuta marža)
- Evaluacija: Brier score vs. bookmaker baseline

## Struktura foldera
- /data/raw        → sirovi CSV fajlovi (Kaggle, kvote)
- /data/processed  → očišćeni dataseti
- /notebooks       → Jupyter notebooci po fazama
- /src             → Python skripte (feature engineering, model)

## Dataseti
- Kaggle: međunarodni rezultati 1872-2024
- football-data.co.uk: SP kvote 2006-2022
- eloratings.net: istorijski Elo rejtinzi

## Pravila
- Svaki rolling feature mora biti izračunat SAMO iz podataka pre datuma utakmice
- Nikakav data leakage
- Komentari na srpskom, kod na engleskom
- Pandas za manipulaciju, scipy.stats.poisson za matricu rezultata