# Fifa_World_Cup_2026_Prediction
# 2026 World Cup Match Prediction

Exploratory data analysis and a first pass at match-outcome prediction using 2026 FIFA World Cup match data.

## Overview

This project works through a full mini data science pipeline on World Cup match data — from raw CSV to a trained classifier — with the eventual (informal) goal of estimating a winner for the Argentina vs. Spain final.

## Dataset

`matches-selected-columns.csv` — one row per match, with columns including:
- `round`, `gameweek`, `dayofweek`, `date`, `start_time`
- `home_team`, `away_team`
- `score`, `home_score`, `away_score`

## What's in the notebook

**1. Data cleaning**
- Filled missing `gameweek` values with the column mean
- Filled missing `home_score` / `away_score` values with the column minimum
- Converted `date` to proper `datetime` and `start_time` to a `time` object

**2. Exploratory Data Analysis**
Six visualizations covering:
- Total goals scored per gameweek
- Home vs. away goals distribution (boxplot)
- Top 10 teams by total goals scored
- Distribution of total goals per match
- Match count by day of week
- Goals scored over time

**3. Team strength ratings + Poisson simulation**
Built per-team goals-for/goals-against stats from the match data, then simulated match outcomes (10,000 Monte Carlo draws) using a Poisson goal model to estimate win/draw/loss probabilities for a given matchup.

**4. Random Forest classifier**
- Engineered a *pre-match* rolling average goals-for feature per team (built chronologically to avoid data leakage — each match only uses stats from matches before it)
- Encoded team names, trained a `RandomForestClassifier` to predict Home Win / Away Win / Draw
- Evaluated with a train/test split and classification report
- Used the trained model to output win probabilities for the Argentina vs. Spain final

## Key limitations

This dataset is small (~100 matches) and covers group-stage matches only, so treat the model's output as an illustrative exercise rather than a genuine forecast:
- No knockout-stage form, head-to-head history, player availability, or rankings data
- Random Forest trained on a handful of matches per team has very little signal to learn from
- Small test set means accuracy scores are noisy; k-fold cross-validation would give a more stable read

## Tech stack

`pandas` · `numpy` · `matplotlib` · `seaborn` · `scikit-learn`

## Possible next steps

- Add knockout-round results as they become available
- Bring in external features (FIFA rankings, squad value, rest days between matches)
- Cross-validate the Random Forest instead of a single train/test split
- Compare Random Forest performance against the simpler Poisson simulation baseline
