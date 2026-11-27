#!/usr/bin/env python3
"""Compare logistic regression with gradient boosting and MLPs (the table in NOTES.md).

  python compare_models.py --user-historical-info user_historical_info.csv

Training dates only, 10 folds grouped by date, same 13 features and imputation as train.py.
"""
import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from train import IMPUTE_FROM, load, score

warnings.filterwarnings('ignore')


def impute(Xtr, Xte):
    """Fill missing values (implied vol by regression, the rest by median) using training-fold statistics."""
    med = Xtr.median()
    ok = Xtr.l_ivd.notna()
    reg = LinearRegression().fit(Xtr.loc[ok, IMPUTE_FROM].fillna(med[IMPUTE_FROM]), Xtr.l_ivd[ok])

    def f(Z):
        Z = Z.copy()
        miss = Z.l_ivd.isna()
        if miss.any():
            Z.loc[miss, 'l_ivd'] = reg.predict(Z.loc[miss, IMPUTE_FROM].fillna(med[IMPUTE_FROM]))
        return Z.fillna(med)
    return f(Xtr), f(Xte)


MODELS = {
    'logistic (chosen)': lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=5000)),
    'GBM depth 2': lambda: HistGradientBoostingClassifier(max_depth=2, learning_rate=0.03, max_iter=300,
                                                          min_samples_leaf=100, l2_regularization=1),
    'GBM depth 3': lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.03, max_iter=300,
                                                          min_samples_leaf=100, l2_regularization=1),
    'GBM depth 5': lambda: HistGradientBoostingClassifier(max_depth=5, learning_rate=0.05, max_iter=300,
                                                          min_samples_leaf=40),
    'GBM monotone': lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.03, max_iter=300,
                                                           min_samples_leaf=100, l2_regularization=1,
                                                           monotonic_cst=[1] * 13),
    'MLP 32-16': lambda: make_pipeline(StandardScaler(), MLPClassifier((32, 16), alpha=1e-2, max_iter=500,
                                                                       early_stopping=True, random_state=0)),
    'MLP 16 strong L2': lambda: make_pipeline(StandardScaler(), MLPClassifier((16,), alpha=1.0, max_iter=1000,
                                                                              random_state=0)),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--user-historical-info', default='user_historical_info.csv')
    a = ap.parse_args()
    X, y, dates = load(a.user_historical_info)
    folds = list(GroupKFold(10).split(X, groups=dates))
    print('%-20s %-7s %-12s %s' % ('model', 'OOF AUC', 'in-sample', 'P / R / score at first threshold with P >= 0.78'))
    for name, make in MODELS.items():
        p = np.zeros(len(y))
        for tr, te in folds:
            a_, b_ = impute(X.iloc[tr], X.iloc[te])
            p[te] = make().fit(a_, y[tr]).predict_proba(b_)[:, 1]
        best = None
        for t in np.unique(np.round(p, 3)):
            f = p > t
            if f.sum() < 30:
                break
            if y[f].mean() >= 0.78:
                P, R = y[f].mean(), (f & y).sum() / y.sum()
                best = '%.2f / %.2f / %.3f' % (P, R, score(P, R))
                break
        full, _ = impute(X, X)
        ins = roc_auc_score(y, make().fit(full, y).predict_proba(full)[:, 1])
        print('%-20s %.3f   %.3f        %s' % (name, roc_auc_score(y, p), ins, best))


if __name__ == '__main__':
    main()
