#!/usr/bin/env python3
"""Predict, for one trading day, which Nifty-50 stocks will move more than 2% from the previous close.

  python predict.py --date YYYYMMDD --historical-info DIR --user-historical-info FILE --out-dir DIR

Logistic regression over stock-level volatility features and market-regime features (see NOTES.md),
with coefficients and threshold in model.json (produced by train.py). Standard library only.
Uses only that day's info file and that day's rows of user_historical_info.csv; both hold only
information public before 09:15.
"""
import argparse
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from features import prob, raw_features  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', required=True)
    ap.add_argument('--historical-info', required=True)
    ap.add_argument('--user-historical-info', default=None)
    ap.add_argument('--out-dir', required=True)
    a = ap.parse_args()

    with open(os.path.join(HERE, 'model.json')) as f:
        model = json.load(f)
    with open(os.path.join(a.historical_info, a.date + '_n50.info.csv'), newline='') as f:
        today = list(csv.DictReader(f))

    user = {}
    if a.user_historical_info and os.path.exists(a.user_historical_info):
        with open(a.user_historical_info, newline='') as f:
            for r in csv.DictReader(f):
                if r['date'] == a.date:
                    user[r['name']] = r

    rows = []
    for r in today:
        try:
            p = prob(raw_features(r, user.get(r['name'])), model)
        except Exception:          # never let one bad row kill the whole date
            p = 0.0
        rows.append(dict(name=r['name'], prediction=p > model['threshold']))

    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, a.date + '.predictions.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['name', 'prediction'])
        w.writeheader()
        w.writerows(rows)


if __name__ == '__main__':
    main()
