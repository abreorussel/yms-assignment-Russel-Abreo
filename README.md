# Approach

**Task.** Predict, before 09:15, whether each Nifty-50 stock will trade more than 2% away from its
previous close during the day. Scoring: precision must exceed 0.60, and the score is a
precision-weighted F-score (β² = 1/3). That makes the problem "maximise recall while keeping
precision safely above 0.60 on 41 unseen days".

**What drives the label.** I first explored the training dates only:
- About 20% of the variance in `moved` depends only on the day. The share of stocks moving ranges from 8% to 94%,
  and there are clear regimes: March–April 58–67%, August 18%. A good model has to recognise calm days.
- Option-implied volatility (`vix / sqrt(252)`) is the best stock-level signal (AUC 0.70). Even its
  top decile moves only 73% of the time, so stock features alone limit recall at the precision bar.

## Data (`prepare_user_historical_info.py` → `user_historical_info.csv`)

Daily bars from Yahoo Finance (yfinance):
- **Per stock** (`<SYMBOL>.NS`, from 2024-06): previous-day return, range, gap and move measure
  `max(H−pC, pC−L)/pC`; 5/20/60-day Parkinson volatility; 20-day close-to-close volatility; the fraction of
  the last 5/20/60 days with a >2% move; and the 20-day average move.
- **Market**: India VIX (level and changes), Nifty 50 and Nifty Bank (returns, range, Parkinson volatility),
  S&P 500, Nasdaq, USD/INR and crude (returns, volatility), and the Nikkei 225 and Hang Seng opening gaps.
- **Results dates** (yfinance `earnings_dates`): `earnings_window` = results due on D or announced on the
  previous trading day. Board-meeting dates are announced to the exchange in advance.

**No look-ahead.** Every feature in a row dated D uses only bars dated strictly before D. US sessions
dated D−1 close around 01:30–02:30 IST on D. The one exception is the **open** of the Nikkei (05:30 IST)
and the Hang Seng (07:00 IST) on day D, both public before 09:15 IST. Their highs, lows and closes on D are
not used. The downloaded series do contain prices for test days, but each day's bar is only used for the
features of *later* days, never for a label. The script also covers the 29 training dates whose info-file
columns are blanked.

## Features and model (`features.py`, `train.py` → `model.json`)

13 features: log implied daily volatility (from the info file), log previous range, log 5/20-day Parkinson
volatility, log 20-day average move, 20-day move rate, log previous move, log India VIX, log Nifty 5-day
Parkinson volatility, log previous Nifty range, mean |Asian opening gap|, mean |US overnight return|, and
the results-window flag.

The model is a standardised **logistic regression** (C = 0.3). Missing implied volatility on the blanked
training dates is imputed from a linear regression on realised volatility and India VIX. Everything else
missing takes the training median. Coefficients, scaling, medians and the threshold are in `model.json`.
`predict.py` uses only the standard library.

The largest standardised coefficients are implied volatility (+0.41), the Asian opening gap (+0.40),
the 20-day average move (+0.32), results (+0.24) and India VIX (+0.16).

## Validation and threshold (training dates only)

- **10-fold cross-validation grouped by date.** All 50 stocks of a day stay in the same fold, because
  stocks on one day are strongly correlated. Out-of-fold AUC is 0.73.
- **Threshold.** Every 41-day draw of training days was simulated (2,000 bootstraps, using out-of-fold
  probabilities). The draws were tilted towards calm days so the base rate is an unlucky but plausible
  value: 2 sd below the mean of random 41-day draws of training days, which is 0.36. This was computed
  from training labels only. The threshold maximises E[score × 1(precision > 0.60)]: a failed submission
  scores 0, so the objective weighs recall against the risk of failing the bar. Result: **threshold 0.655**.
  - Out-of-fold, all training days: precision 0.78, recall 0.29, score 0.74.
  - Calm 41-day draws: median precision 0.74, recall 0.23, P(pass) ≈ 0.98, expected score 0.67.
- **Walk-forward** (train on earlier days, predict the next 10): much weaker. A model trained only on
  the volatile March–May regime has to predict the calm August regime. The test days are drawn at random
  from the whole period, so grouped cross-validation is the relevant estimate. Walk-forward shows the
  model is sensitive to which regimes it has seen.
- The grader in practice mode on the training dates (in-sample): precision 0.80, recall 0.30, score 0.75.

The test labels and test dates in `evaluator/data/` were not used for any feature, model or threshold choice.

## What I tried that did not help

These were all checked with date-grouped cross-validation:
- **VIX z-score, `vix/vix_1m` and the traded-value ratio** (`pvolume/mddv_20`): no gain.
- **Interactions** between stock volatility and market regime: no gain.
- **Count of reporting stocks on the day:** no gain.
- **Stronger or weaker regularisation** (C = 0.05–3): no change.
- **More flexible models.** I compared them on the same features and the same 10 date-grouped folds.
  The operating point is the first threshold that reaches 0.78 out-of-fold precision:

  Reproduced by `compare_models.py`.

  | Model | Out-of-fold AUC | In-sample AUC | Precision / Recall / Score |
  |---|---|---|---|
  | **Logistic regression (chosen)** | **0.732** | 0.742 | 0.78 / **0.29** / **0.743** |
  | Gradient boosting, depth 2 | 0.718 | 0.773 | 0.78 / 0.29 / 0.738 |
  | Gradient boosting, depth 3 | 0.708 | 0.797 | 0.78 / 0.27 / 0.729 |
  | Gradient boosting, depth 5 | 0.696 | 0.898 | 0.78 / 0.25 / 0.711 |
  | Gradient boosting, depth 3, monotone constraints | 0.721 | 0.769 | 0.78 / 0.29 / 0.739 |
  | MLP (32-16) | 0.725 | 0.741 | 0.78 / 0.24 / 0.707 |
  | MLP (16, strong L2) | 0.727 | 0.752 | 0.78 / 0.27 / 0.728 |

  Gradient boosting is sklearn's HistGradientBoosting, the same algorithm family as LightGBM and XGBoost.
  The more flexible the model, the wider the gap between in-sample and out-of-fold AUC. Three reasons:
  - Stocks on the same day are strongly correlated, so the market-regime signal has only about 97
    independent examples (days), not 4,850.
  - The relationship is close to additive and monotone in log-volatility terms. Monotone constraints
    help the trees, which supports this.
  - A single threshold must carry over to calmer days, which favours the smoother, better-calibrated
    probabilities of a linear model.
- **Picking the threshold at a fixed 0.68 validation precision:** calm-day draws failed the 0.60 bar
  about 38% of the time. That is why the threshold is chosen by the risk-aware objective instead.

## Reproduce

```
pip install -r requirements.txt
python prepare_user_historical_info.py --out user_historical_info.csv
python train.py --user-historical-info user_historical_info.csv      # writes model.json
python compare_models.py --user-historical-info user_historical_info.csv   # the model comparison table
python evaluator/run_evaluation.py .                                  # practice run on training dates
```
