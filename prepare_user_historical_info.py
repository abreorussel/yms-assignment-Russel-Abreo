#!/usr/bin/env python3
"""Build user_historical_info.csv: extra per-(date, name) features for predict.py.

  pip install yfinance pandas numpy
  python prepare_user_historical_info.py --out my_submission/user_historical_info.csv

Sources (all from Yahoo Finance via yfinance, daily bars):
  - each Nifty-50 stock (<SYMBOL>.NS) from 2024-06-01, for realised volatility and past move rates
  - Nifty 50, Nifty Bank, India VIX, S&P 500, Nasdaq, Nikkei 225, Hang Seng, USD/INR, crude oil
  - scheduled results dates per stock (yfinance earnings_dates), when available

NO LOOK-AHEAD. For a row dated D:
  - every stock / Indian index / US / FX / crude feature uses only bars dated strictly before D
    (US markets close around 01:30-02:30 IST on D, so the US bar dated D-1 is known by 09:15 IST);
  - the only bars dated D that are used are the OPEN of the Nikkei (opens 05:30 IST) and the
    Hang Seng (opens 07:00 IST), both public before 09:15 IST. Their highs, lows and closes are not used;
  - results dates are board-meeting dates announced to the exchange weeks in advance.
Day D's own prices are downloaded (they are in the same series) but are never used for a row dated D.

Raw downloads are cached in ./yf_cache/ so reruns are fast; delete it to refresh.
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

START = '2024-06-01'          # enough history for 60-day windows before 2026-03-04
END = '2026-11-05'            # yfinance end is exclusive; covers 2026-10-30

# Yahoo symbols to try for each NSE symbol (first that returns data wins)
SPECIAL = {
    'M&M': ['M&M.NS'],
    'BAJAJ-AUTO': ['BAJAJ-AUTO.NS'],
    'TMPV': ['TMPV.NS', 'TATAMOTORS.NS'],
    'ETERNAL': ['ETERNAL.NS', 'ZOMATO.NS'],
}

MARKET = {
    'nifty': '^NSEI', 'banknifty': '^NSEBANK', 'indiavix': '^INDIAVIX',
    'spx': '^GSPC', 'ndx': '^IXIC', 'nikkei': '^N225', 'hsi': '^HSI',
    'usdinr': 'INR=X', 'crude': 'CL=F',
}
ASIA_OPEN_OK = {'nikkei', 'hsi'}   # their open on day D is known before 09:15 IST


def fetch(sym, cache):
    path = os.path.join(cache, sym.replace('^', '_').replace('=', '_').replace('&', '_') + '.csv')
    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df if len(df) else None
    import yfinance as yf
    for attempt in range(3):
        try:
            df = yf.Ticker(sym).history(start=START, end=END, auto_adjust=True)
            break
        except Exception as e:  # network hiccup: retry
            print('  retry', sym, e)
            time.sleep(2)
    else:
        df = pd.DataFrame()
    if len(df):
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        df = df[['Open', 'High', 'Low', 'Close']].dropna()
        df = df[df['Close'] > 0]
    df.to_csv(path)
    return df if len(df) else None


def fetch_earnings(sym, cache):
    """Set of results dates (as date) for a Yahoo symbol; empty if unavailable."""
    path = os.path.join(cache, 'earn_' + sym.replace('&', '_') + '.txt')
    if os.path.exists(path):
        return {pd.Timestamp(x) for x in open(path).read().split()}
    out = set()
    try:
        import yfinance as yf
        ed = yf.Ticker(sym).get_earnings_dates(limit=24)
        if ed is not None and len(ed):
            idx = pd.to_datetime(ed.index)
            if idx.tz is not None:
                idx = idx.tz_convert('Asia/Kolkata').tz_localize(None)
            out = {t.normalize() for t in idx}
    except Exception as e:
        print('  no earnings dates for', sym, '-', e)
    with open(path, 'w') as f:
        f.write('\n'.join(t.strftime('%Y-%m-%d') for t in sorted(out)))
    return out


def stock_features(df):
    """Per-bar features computed from bars up to and including that bar.
    They are later shifted so that a row dated D only sees bars dated < D."""
    o, h, l, c = df['Open'], df['High'], df['Low'], df['Close']
    pc = c.shift(1)
    ret = np.log(c / pc)
    pk = np.log(h / l) ** 2 / (4 * np.log(2))                      # Parkinson daily variance
    maxmv = np.maximum(h - pc, pc - l) / pc * 100                   # the task's move measure, in %
    moved = (maxmv > 2).astype(float).where(pc.notna())
    f = pd.DataFrame(index=df.index)
    f['ret1'] = ret * 100
    f['range1'] = (h - l) / pc * 100
    f['gap1'] = (o / pc - 1) * 100
    f['maxmove1'] = maxmv
    for w in (5, 20, 60):
        f[f'pk_vol{w}'] = np.sqrt(pk.rolling(w, min_periods=max(3, w // 2)).mean()) * 100   # daily %
        f[f'move_rate{w}'] = moved.rolling(w, min_periods=max(3, w // 2)).mean()
    f['cc_vol20'] = ret.rolling(20, min_periods=10).std() * 100
    f['maxmove_avg20'] = maxmv.rolling(20, min_periods=10).mean()
    f['ret5'] = (c / c.shift(5) - 1) * 100
    f['ret20'] = (c / c.shift(20) - 1) * 100
    return f


def market_features(name, df):
    c, h, l = df['Close'], df['High'], df['Low']
    ret = np.log(c / c.shift(1)) * 100
    f = pd.DataFrame(index=df.index)
    f[f'{name}_ret1'] = ret
    f[f'{name}_ret5'] = (c / c.shift(5) - 1) * 100
    f[f'{name}_vol20'] = ret.rolling(20, min_periods=10).std()
    if name in ('nifty', 'banknifty'):
        f[f'{name}_range1'] = (h - l) / c.shift(1) * 100
        pk = np.log(h / l) ** 2 / (4 * np.log(2))
        f[f'{name}_pk_vol5'] = np.sqrt(pk.rolling(5, min_periods=3).mean()) * 100
    if name == 'indiavix':
        f = pd.DataFrame(index=df.index)
        f['indiavix'] = c
        f['indiavix_chg1'] = (c / c.shift(1) - 1) * 100
        f['indiavix_chg5'] = (c / c.shift(5) - 1) * 100
        f['indiavix_rel20'] = c / c.rolling(20, min_periods=10).mean()
    return f


def asof_before(feat, dates):
    """For each date D, the feature row of the last bar dated strictly before D."""
    feat = feat.sort_index()
    pos = feat.index.searchsorted(dates, side='left') - 1
    out = feat.iloc[np.clip(pos, 0, None)].copy()
    out.iloc[pos < 0] = np.nan
    out.index = dates
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dates', default='data/all_dates.txt')
    ap.add_argument('--names', default='data/n50_names.txt')
    ap.add_argument('--out', default='user_historical_info.csv')
    ap.add_argument('--cache', default='yf_cache')
    ap.add_argument('--no-earnings', action='store_true')
    a = ap.parse_args()
    os.makedirs(a.cache, exist_ok=True)

    dates_s = open(a.dates).read().split()
    names = open(a.names).read().split()
    dates = pd.DatetimeIndex(pd.to_datetime(dates_s, format='%Y%m%d'))

    # ---------- market-wide features (same for all stocks on a date) ----------
    mkt = pd.DataFrame(index=dates)
    for name, sym in MARKET.items():
        df = fetch(sym, a.cache)
        if df is None:
            print('WARNING: no data for', name, sym)
            continue
        mkt = mkt.join(asof_before(market_features(name, df), dates))
        if name in ASIA_OPEN_OK:
            # open of day D (known before 09:15 IST) vs the last close before D
            prev_close = asof_before(df[['Close']], dates)['Close']
            today_open = df['Open'].reindex(dates)          # NaN if that market was shut on D
            mkt[f'{name}_gap'] = (today_open / prev_close - 1) * 100
    # US overnight move: the US bar dated D-1 (or the last before D) closes before 09:15 IST on D
    mkt['us_abs_ret1'] = mkt[['spx_ret1', 'ndx_ret1']].abs().mean(axis=1) if 'spx_ret1' in mkt else np.nan
    mkt['asia_abs_gap'] = mkt[[c for c in ('nikkei_gap', 'hsi_gap') if c in mkt]].abs().mean(axis=1)

    # ---------- per-stock features ----------
    rows = []
    n_earn = 0
    for n in names:
        syms = SPECIAL.get(n, [n + '.NS'])
        df, used = None, None
        for s in syms:
            df = fetch(s, a.cache)
            if df is not None:
                used = s
                break
        if df is None:
            print('WARNING: no price data for', n)
            sf = pd.DataFrame(index=dates)
        else:
            sf = asof_before(stock_features(df), dates)
        if not a.no_earnings and used:
            ed = fetch_earnings(used, a.cache)
            # results due on D, or announced on the previous trading day (often after the close)
            prev_td = asof_before(pd.DataFrame({'d': df.index}, index=df.index), dates)['d'] if df is not None else None
            flag = []
            for i, d in enumerate(dates):
                p = prev_td.iloc[i] if prev_td is not None else None
                flag.append(int(d in ed or (p is not None and pd.notna(p) and p in ed)))
            sf['earnings_window'] = flag
            n_earn += sum(flag)
        sf.insert(0, 'name', n)
        rows.append(sf)
    stocks = pd.concat(rows)
    stocks.index.name = 'dt'

    out = stocks.join(mkt, on='dt').reset_index()
    out.insert(0, 'date', out.pop('dt').dt.strftime('%Y%m%d'))
    out['weekday'] = pd.to_datetime(out['date']).dt.weekday
    out = out.sort_values(['date', 'name']).round(5)
    out.to_csv(a.out, index=False)

    print('wrote %s: %d rows (%d dates x %d names), %d columns' % (a.out, len(out), len(dates), len(names), out.shape[1]))
    print('earnings-window rows:', n_earn)
    miss = out.isna().mean().sort_values(ascending=False)
    print('columns with missing values (fraction):')
    print(miss[miss > 0].round(3).to_string() if (miss > 0).any() else '  none')


if __name__ == '__main__':
    main()
