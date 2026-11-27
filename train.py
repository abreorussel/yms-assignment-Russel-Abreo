#!/usr/bin/env python3
"""Fit the model and choose its threshold using TRAINING dates only; writes model.json.

  python train.py --user-historical-info my_submission/user_historical_info.csv

Reads data/historical_info/, data/train_labels.csv and the user file. Never reads evaluator/data/.
Needs numpy, pandas, scikit-learn (predict.py itself needs only the standard library).
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

from features import FEATURES, raw_features

HERE = os.path.dirname(os.path.abspath(__file__))
C_REG = 0.3
IMPUTE_FROM = ['l_pk5', 'l_pk20', 'l_mm20', 'l_ivix']
N_BOOT, N_DAYS = 2000, 41


def score(P, R):
    return np.sqrt(1 / (0.75 / P + 0.25 / R)) if P > 0 and R > 0 else 0.0


def load(user_path):
    info = pd.concat([pd.read_csv(f, dtype=str) for f in sorted(glob.glob(os.path.join(HERE, 'data/historical_info/*.csv')))])
    lab = pd.read_csv(os.path.join(HERE, 'data/train_labels.csv'), dtype={'date': str})
    user = pd.read_csv(user_path, dtype=str)
    info_d = {(r['date'], r['name']): r for r in info.to_dict('records')}
    user_d = {(r['date'], r['name']): r for r in user.to_dict('records')}
    rows = [raw_features(info_d.get((dt, n)), user_d.get((dt, n))) for dt, n in zip(lab.date, lab.name)]
    X = pd.DataFrame(rows, columns=FEATURES).astype(float)
    return X, lab.moved.astype(bool).values, lab.date.values


def fit(X, y):
    """Fit imputation + scaler + logistic regression on (X, y). Returns a JSON-able model dict."""
    med = X.median()
    ok = X.l_ivd.notna()
    Ximp = X[IMPUTE_FROM].fillna(med[IMPUTE_FROM])
    lr_imp = LinearRegression().fit(Ximp[ok], X.l_ivd[ok])
    model = {'medians': {k: float(med[k]) for k in FEATURES},
             'ivd_impute': {'features': IMPUTE_FROM, 'intercept': float(lr_imp.intercept_),
                            'coef': [float(c) for c in lr_imp.coef_]}}
    Xf = X.copy()
    Xf.loc[~ok, 'l_ivd'] = lr_imp.predict(Ximp[~ok])
    Xf = Xf.fillna(med)
    mu, sd = Xf.mean(), Xf.std().replace(0, 1)
    lr = LogisticRegression(C=C_REG, max_iter=5000).fit((Xf - mu) / sd, y)
    model.update(features=FEATURES, mean=[float(v) for v in mu], std=[float(v) for v in sd],
                 coef=[float(c) for c in lr.coef_[0]], intercept=float(lr.intercept_[0]))
    return model


def predict_df(X, model):
    Xf = X.copy()
    imp = model['ivd_impute']
    med = pd.Series(model['medians'])
    miss = Xf.l_ivd.isna()
    if miss.any():
        Z = Xf.loc[miss, imp['features']].fillna(med[imp['features']])
        Xf.loc[miss, 'l_ivd'] = imp['intercept'] + Z.values @ np.array(imp['coef'])
    Xf = Xf.fillna(med)
    z = model['intercept'] + (((Xf - model['mean']) / model['std']).values @ np.array(model['coef']))
    return 1 / (1 + np.exp(-z))


def bootstrap(p, y, dates, thresholds, tilt, rng):
    """For each threshold: P(pass) and E[score * pass] over random 41-day samples."""
    days = np.unique(dates)
    idx = {d: np.where(dates == d)[0] for d in days}
    rate = np.array([y[idx[d]].mean() for d in days])
    w = np.exp(-tilt * rate); w /= w.sum()
    samples = [rng.choice(len(days), N_DAYS, p=w) for _ in range(N_BOOT)]
    out = []
    for t in thresholds:
        flag = p > t
        tp_d = np.array([(flag[idx[d]] & y[idx[d]]).sum() for d in days])
        fl_d = np.array([flag[idx[d]].sum() for d in days])
        pos_d = np.array([y[idx[d]].sum() for d in days])
        P = tp_d[samples].sum(1) / np.maximum(fl_d[samples].sum(1), 1)
        R = tp_d[samples].sum(1) / pos_d[samples].sum(1)
        S = np.sqrt(1 / (0.75 / np.maximum(P, 1e-9) + 0.25 / np.maximum(R, 1e-9)))
        out.append((t, (P > 0.6).mean(), (S * (P > 0.6)).mean(), np.median(P), np.median(R)))
    return pd.DataFrame(out, columns=['thr', 'p_pass', 'exp_score', 'med_P', 'med_R']), (w * rate).sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--user-historical-info', default='my_submission/user_historical_info.csv')
    ap.add_argument('--out', default=os.path.join(HERE, 'model.json'))
    a = ap.parse_args()
    X, y, dates = load(a.user_historical_info)
    print('training rows %d, days %d, move rate %.3f' % (len(y), len(np.unique(dates)), y.mean()))

    # 1) out-of-fold probabilities, folds grouped by date
    p = np.zeros(len(y))
    for tr, te in GroupKFold(10).split(X, groups=dates):
        p[te] = predict_df(X.iloc[te], fit(X.iloc[tr], y[tr]))
    print('OOF AUC %.3f' % roc_auc_score(y, p))

    # 2) threshold: maximise expected (score if pass else 0) on calm-tilted 41-day samples.
    #    "Calm" = an unlucky 41-day draw: 2 sd below the mean base rate of random 41-day draws
    #    of TRAINING days (computed here from training labels only).
    day_rate = np.array([y[dates == d].mean() for d in np.unique(dates)])
    r2 = np.random.default_rng(1)
    draws = np.array([day_rate[r2.choice(len(day_rate), N_DAYS, replace=False)].mean() for _ in range(20000)])
    CALM_RATE = draws.mean() - 2 * draws.std()
    print('calm scenario base rate (train-only, mean - 2 sd of 41-day draws): %.3f' % CALM_RATE)
    rng = np.random.default_rng(0)
    th = np.round(np.arange(0.45, 0.80, 0.005), 3)
    tilt = 0.0
    for t_ in np.arange(0, 5, 0.05):            # tilt so the sampled base rate equals CALM_RATE
        days = np.unique(dates); rate = np.array([y[dates == d].mean() for d in days])
        w = np.exp(-t_ * rate); w /= w.sum()
        if (w * rate).sum() <= CALM_RATE:
            tilt = t_; break
    calm, br = bootstrap(p, y, dates, th, tilt, rng)
    plain, _ = bootstrap(p, y, dates, th, 0.0, rng)
    best = calm.loc[calm.exp_score.idxmax()]
    thr = float(best.thr)
    f = p > thr
    P, R = y[f].mean(), (f & y).sum() / y.sum()
    print('\nthreshold %.3f' % thr)
    print('  OOF on all training days : precision %.3f recall %.3f score %.3f' % (P, R, score(P, R)))
    pr = plain[plain.thr == thr].iloc[0]
    print('  41-day samples, as is    : P(pass) %.2f, median precision %.3f, recall %.3f' % (pr.p_pass, pr.med_P, pr.med_R))
    print('  41-day samples, calm %.3f: P(pass) %.2f, median precision %.3f, recall %.3f, E[score*pass] %.3f'
          % (br, best.p_pass, best.med_P, best.med_R, best.exp_score))
    print('\n  thr   p_pass(calm)  E[score*pass](calm)  medP  medR')
    for _, r in calm.iloc[::4].iterrows():
        print('  %.3f   %.2f          %.3f              %.3f %.3f' % (r.thr, r.p_pass, r.exp_score, r.med_P, r.med_R))

    # 3) walk-forward check with the chosen threshold: train on all earlier days, predict next 10
    days = np.unique(dates)
    tp = fl = pos = 0
    for start in range(40, len(days), 10):
        trd, ted = days[:start], days[start:start + 10]
        tr, te = np.isin(dates, trd), np.isin(dates, ted)
        pp = predict_df(X[te], fit(X[tr], y[tr])) > thr
        tp += (pp & y[te]).sum(); fl += pp.sum(); pos += y[te].sum()
    print('\nwalk-forward (days 41-97): precision %.3f recall %.3f score %.3f' % (tp / fl, tp / pos, score(tp / fl, tp / pos)))

    # 4) final model on all training days
    model = fit(X, y)
    model['threshold'] = thr
    with open(a.out, 'w') as fh:
        json.dump(model, fh, indent=1)
    print('\ncoefficients (standardised):')
    for k, c in sorted(zip(FEATURES, model['coef']), key=lambda kc: -abs(kc[1])):
        print('  %-12s %+.3f' % (k, c))
    print('wrote', a.out)


if __name__ == '__main__':
    main()
