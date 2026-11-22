"""Market-wide features for every order-book snapshot, from public Yahoo Finance bars (hourly and daily).
Strictly causal: at KRX time t an hourly bar is used only if it ended at least 60 s before t (bar start + 1 h <= t - 60 s;
the short last bar of a session therefore counts as ending one hour after its start); a daily US bar is used only if
its US date is before the KRX date (the US session closes before the KRX open).
Features (24)
  KRX-hours peers, hourly - KOSPI 200, Samsung Electronics (005930), Nikkei 225: return and high/low range of the last
      finished bar, return since the day's first bar opened; missing (NaN) until the first bar of the day has finished
  around-the-clock, hourly - Nasdaq-100 and S&P 500 futures, USD/KRW: return and range of the last finished bar,
      return over the last 24 h
  previous US session, daily - Philadelphia semiconductor index (return, range), VIX (log level, change),
      Micron and NVIDIA (return)
All returns and ranges are natural-log ratios.
usage: python market_feat.py fetch | build   (fetch: data/market/*.json from Yahoo Finance; build: data/feat_mkt/<day>.npz)
"""
import json, os, sys, time, datetime as dt
import numpy as np
import common as C

MD = os.path.join(C.DATA, 'market'); FD = os.path.join(C.DATA, 'feat_mkt'); LAG = 60
SYMS = dict(ks200='^KS200', ss='005930.KS', nk='^N225', nq='NQ=F', es='ES=F', fx='KRW=X', sox='^SOX', vix='^VIX', mu='MU', nvda='NVDA')
PEERS, H24, DAILY = ('ks200', 'ss', 'nk'), ('nq', 'es', 'fx'), ('sox', 'vix', 'mu', 'nvda')
NAMES = ([f'{k}_{f}' for k in PEERS for f in ('ret1h', 'rng1h', 'retday')] + [f'{k}_{f}' for k in H24 for f in ('ret1h', 'rng1h', 'ret24h')]
         + ['sox_ret1d', 'sox_rng1d', 'vix_lvl', 'vix_chg', 'mu_ret1d', 'nvda_ret1d'])


def fname(sym, iv): return os.path.join(MD, sym.replace('^', 'I_').replace('=', '_').replace('.', '_') + f'_{iv}.json')


def fetch():
    import urllib.request, urllib.parse
    os.makedirs(MD, exist_ok=True)
    p1 = int(dt.datetime(2026, 6, 1, tzinfo=dt.timezone.utc).timestamp()); p2 = int(dt.datetime(2026, 11, 12, tzinfo=dt.timezone.utc).timestamp())
    for s in SYMS.values():
        for iv in ('60m', '1d'):
            url = f'https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(s)}?period1={p1}&period2={p2}&interval={iv}&includePrePost=false'
            r = json.loads(urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'}), timeout=30).read())
            json.dump(r['chart']['result'][0], open(fname(s, iv), 'w')); time.sleep(0.6)


def bars(key, iv):
    r = json.load(open(fname(SYMS[key], iv))); q = r['indicators']['quote'][0]
    a = {k: np.array([np.nan if v is None else v for v in q[k]], float) for k in ('open', 'high', 'low', 'close')}
    t = np.array(r['timestamp'], np.int64); ok = np.isfinite(a['open']) & np.isfinite(a['close']) & np.isfinite(a['high']) & np.isfinite(a['low'])
    return t[ok], {k: v[ok] for k, v in a.items()}


def day_features(d, ts_ns, H, Dl):
    t = ts_ns / 1e9 - LAG; n = len(t); out = np.full((n, len(NAMES)), np.nan, np.float32); c = 0
    day0 = dt.datetime.strptime(d, '%Y%m%d').replace(tzinfo=dt.timezone(dt.timedelta(hours=9))).timestamp()   # 00:00 KST
    for k in PEERS + H24:
        st, a = H[k]; j = np.searchsorted(st + 3600, t, side='right') - 1; ok = j >= 0; jj = np.maximum(j, 0)
        r1 = np.log(a['close'][jj] / a['open'][jj]); rg = np.log(a['high'][jj] / a['low'][jj])
        if k in PEERS:
            today = ok & (st[jj] >= day0); f0 = np.searchsorted(st, day0)                      # first bar of the KRX date
            third = np.where(today, np.log(a['close'][jj] / a['open'][min(f0, len(st) - 1)]), np.nan)
            ok = today
        else:
            j24 = np.searchsorted(st + 3600, t - 86400, side='right') - 1
            third = np.where(j24 >= 0, np.log(a['close'][jj] / a['close'][np.maximum(j24, 0)]), np.nan)
        out[:, c] = np.where(ok, r1, np.nan); out[:, c + 1] = np.where(ok, rg, np.nan); out[:, c + 2] = third; c += 3
    D = dt.date(int(d[:4]), int(d[4:6]), int(d[6:]))
    for k in DAILY:
        st, a = Dl[k]; dates = np.array([dt.datetime.fromtimestamp(x, dt.timezone.utc).date() for x in st]); i = np.searchsorted(dates, D) - 1   # last US date < KRX date
        if i < 1: c += {'sox': 2, 'vix': 2}.get(k, 1); continue
        if k == 'sox': out[:, c] = np.log(a['close'][i] / a['close'][i - 1]); out[:, c + 1] = np.log(a['high'][i] / a['low'][i]); c += 2
        elif k == 'vix': out[:, c] = np.log(a['close'][i]); out[:, c + 1] = np.log(a['close'][i] / a['close'][i - 1]); c += 2
        else: out[:, c] = np.log(a['close'][i] / a['close'][i - 1]); c += 1
    assert c == len(NAMES)
    return out


def build():
    os.makedirs(FD, exist_ok=True); H = {k: bars(k, '60m') for k in PEERS + H24}; Dl = {k: bars(k, '1d') for k in DAILY}
    TR, VA, TE = C.split_days(); cover = []
    for d in TR + VA + TE:
        z = C.load_day(d); X = day_features(d, z['ts'].astype(np.int64), H, Dl)
        np.savez_compressed(os.path.join(FD, f'{d}.npz'), X=X, names=np.array(NAMES)); cover.append(float(np.isfinite(X).mean()))
    print('days', len(cover), 'mean share of non-missing values', round(float(np.mean(cover)), 3), flush=True)


if __name__ == '__main__':
    {'fetch': fetch, 'build': build}[sys.argv[1]]()
