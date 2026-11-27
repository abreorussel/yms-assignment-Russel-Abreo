"""Feature construction shared by train.py and predict.py (standard library only).

Every input is known before 09:15 on the row's date:
  - info file columns (previous close / VIX as of the previous session), and
  - user_historical_info.csv columns, built by prepare_user_historical_info.py from bars dated
    strictly before the date (plus the Nikkei / Hang Seng opens, which happen before 09:15 IST).
"""
import math

FEATURES = [
    'l_ivd',        # log option-implied daily vol (vix / sqrt 252), from the info file
    'l_rng1',       # log previous-day high-low range, % of the close before it
    'l_pk5',        # log 5-day Parkinson (range-based) daily vol
    'l_pk20',       # log 20-day Parkinson daily vol
    'l_mm20',       # log mean over 20 days of the task's own move measure max(H-pC, pC-L)/pC
    'move_rate20',  # fraction of the last 20 days on which the stock moved > 2%
    'l_mm1',        # log previous day's move measure
    'l_ivix',       # log India VIX (previous close)            -- market regime
    'l_npk5',       # log 5-day Parkinson vol of Nifty 50        -- market regime
    'l_nr1',        # log previous-day Nifty range               -- market regime
    'asia',         # mean |opening gap| of Nikkei and Hang Seng on the day (both open before 09:15 IST)
    'us',           # mean |overnight return| of S&P 500 and Nasdaq
    'earn',         # 1 if results are due today or were announced on the previous trading day
]


def num(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _log(v):
    return math.log(max(v, 1e-3)) if v is not None else None


def raw_features(info_row, user_row):
    """Dict of FEATURES (value or None) for one (date, stock)."""
    u = user_row or {}
    g = lambda k: num(u.get(k))
    vix = num(info_row.get('vix')) if info_row else None
    return {
        'l_ivd': _log(vix / math.sqrt(252)) if vix else None,
        'l_rng1': _log(g('range1')),
        'l_pk5': _log(g('pk_vol5')),
        'l_pk20': _log(g('pk_vol20')),
        'l_mm20': _log(g('maxmove_avg20')),
        'move_rate20': g('move_rate20'),
        'l_mm1': _log(g('maxmove1')),
        'l_ivix': _log(g('indiavix')),
        'l_npk5': _log(g('nifty_pk_vol5')),
        'l_nr1': _log(g('nifty_range1')),
        'asia': g('asia_abs_gap'),
        'us': g('us_abs_ret1'),
        'earn': g('earnings_window'),
    }


def fill(feat, model):
    """Fill missing values: implied vol from its regression on realised vol, everything else by median."""
    f = dict(feat)
    if f['l_ivd'] is None:
        imp = model['ivd_impute']
        xs = [f.get(k) if f.get(k) is not None else model['medians'][k] for k in imp['features']]
        f['l_ivd'] = imp['intercept'] + sum(c * x for c, x in zip(imp['coef'], xs))
    for k in FEATURES:
        if f[k] is None:
            f[k] = model['medians'][k]
    return f


def prob(feat, model):
    f = fill(feat, model)
    z = model['intercept']
    for k, c, mu, sd in zip(FEATURES, model['coef'], model['mean'], model['std']):
        z += c * (f[k] - mu) / sd
    return 1.0 / (1.0 + math.exp(-z))
