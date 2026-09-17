# Survivor ML

Two models that score every remaining player, episode by episode, across all US seasons of Survivor:

- **Win model**: probability a player wins the season
- **Elimination model**: probability a player goes home this episode

Live site: **https://victoriaritvo.com/survivor-ml/**

Both beat random guessing by roughly 2x. Under expanding-window cross-validation, the win model's top pick is the eventual winner 22% of the time (vs 10% for a random guess) and it ranks the winner 4.9th out of the remaining players on average (vs 6.4). The elimination model's top pick goes home 21% of the time (vs 10%). (As of S50. Numbers will shift a little as seasons get added.)

All of the data comes from [doehm/survivoR](https://github.com/doehm/survivoR), which packages up Survivor data by season, episode, and castaway.


## Setup

```bash
uv sync
```

## Running things

Rebuild everything the site reads. Loads the data, builds the feature table, runs both models, writes one JSON per season:

```bash
uv run python main.py                  # use the local data/survivoR.xlsx
uv run python main.py --fetch-data     # download the latest spreadsheet first
```

 The first run needs `--fetch-data` to download the current spreadsheet into `data/`.


Run a single model and print its metrics:

```bash
uv run python -m src.models.win
uv run python -m src.models.elimination
```

Either model also takes:

```bash
--tune          # grid search the hyperparameters
--select        # greedy forward feature selection
```

and the win model additionally takes `--predict 50` to train on everything before season 50 and print its top picks for it.

Front end:

```bash
cd app && npm install && npm run dev
```

## Structure

```
main.py                   # rebuild all season JSON (this is what CI runs)
src/                      # the code to train / test the models
  models/                 # models (elimination.py & win.py)
app/                      # Vite front end for the dashboard
```

The modeling table is one row per (season, episode, player still in game). Every feature is lagged so nothing leaks from the episode being predicted, and within each episode the predicted probabilities are normalized to sum to 1.

Reported accuracy comes from expanding-window temporal cross-validation (always training on earlier seasons and testing on later ones).

The JSON the app reads is built differently: each season is scored leave-one-season-out. For the current season that's the same thing as training on prior seasons only, so those numbers really are forecasts. For past seasons the training set includes later ones, which stabilizes the early-era trajectories the site displays but makes them descriptive rather than predictive.

## Automation

Two GitHub Actions workflows:

- `refresh-data.yml` — Tuesdays and Saturdays, downloads the latest spreadsheet, regenerates the season JSON, commits it if anything changed, then triggers a deploy
- `deploy.yml` — builds `app/` and publishes to GitHub Pages on every push to `main`