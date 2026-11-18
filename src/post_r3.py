"""Round 3, 5 s direction: combination and post-processing of the members' VAL probabilities. TEST is not read here.
Design fixed before running.
Rows   : the VAL rows every member in the pool can score (keys day x row), so all candidates are compared on the same
         rows.  The round-2 winner (gbm + mt + tcn, 5-seed bags, flat multiplier 1.1) is re-scored on these rows.
Pool   : gbm, logreg (exp_v2.py); bags base / mt / tcn of all seeds present (deep_v2.py seeds 0-4 + deep_r3.py 5-9);
         the round-3 variants with 5 seeds (deep_r3.py R2); the round-3 GBMs (gbm_r3.py).  A bag is the plain mean of
         its seeds' probabilities.
         pool 'main' leaves out TLOB: one TLOB prediction costs 29x a conv+GRU one, too slow for ~4.5 snapshots/s on
         a CPU. Note: this rule was set after TLOB's TEST result (lob_models.py) was known. pool 'with_tlob' adds
         the TLOB bag (3 seeds) and is reported for reference only.
Methods (each scored on VAL macro-F1)
  avg    : plain mean of every subset of up to 5 members, flat-class multiplier m from (0.7 ... 1.3)  [as in round 2]
  cls    : the 5 best 'avg' subsets with a multiplier per class: flat m in 0.80-1.40 (step 0.05) and up/down
           ratio in 0.90-1.10 (step 0.025)
  smooth : the 5 best 'avg' subsets with causal exponential smoothing of the probabilities along consecutive rows of
           the same segment, p~_t = (1 - a) p_t + a p~_{t-1}, a in (0.3, 0.5, 0.7, 0.8, 0.9), then m
  stack  : a model on the members' log-probabilities (+ time of day, spread, short-term volatility):
           multinomial logistic regression and a small GBM; scores are leave-one-day-out over the 12 VAL days, so a
           VAL score never comes from a model that saw that day; then a multiplier per class
           Changed after the dry run on the round-2 members (before any round-3 result): VAL is 62.5 % 'flat', so the
           class-balanced stacker predicts 'flat' far less than the members and its best flat multiplier lay at the
           edge of the grid (1.3, VAL 0.511). Each stacker is therefore also fitted without class weights ('_uw',
           calibrated VAL probabilities), and every stacker gets the per-class multiplier search over a wide grid
           (flat 0.40-2.00 step 0.05, up/down ratio 0.90-1.10 step 0.025). The balanced stackers are kept as designed.
Winner : highest VAL macro-F1.  Split-half check: the whole selection is repeated on VAL days 1-6 and scored on days
         7-12 and the reverse, next to the round-2 winner on the same halves.
usage: python post_r3.py main|with_tlob -> results/r3_post.jsonl
"""
import glob, json, os, re, sys, time
from itertools import combinations
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
import common as C
import exp_v2 as E

OUT = E.OUT; SMOKE = os.environ.get('R3_POST_SMOKE')         # SMOKE: round-2 files only, output to the given file
RES = SMOKE or os.path.join(OUT, 'r3_post.jsonl')
KEY = lambda z: z['day'].astype(np.int64) * 1_000_000 + z['row'].astype(np.int64)
MS = E.MS; MF = np.round(np.arange(0.80, 1.401, 0.05), 3); UD = np.round(np.arange(0.90, 1.101, 0.025), 3); AL = (0.3, 0.5, 0.7, 0.8, 0.9)
MFW = np.round(np.arange(0.40, 2.001, 0.05), 3)              # wide flat grid for the stackers
KMAX = 5                                                   # largest subset size tried


POOL = sys.argv[1] if len(sys.argv) > 1 else 'main'


def log(rec):
    rec = dict(pool=POOL, **rec); print(json.dumps(rec)[:300], flush=True)
    with open(RES, 'a') as f: f.write(json.dumps(rec) + '\n')


def mf1(y, p):
    """macro-F1 from a confusion matrix (fast)"""
    cm = np.bincount(y * 3 + p, minlength=9).reshape(3, 3); tp = np.diag(cm).astype(float)
    return float(np.mean(2 * tp / np.maximum(cm.sum(0) + cm.sum(1), 1)))


def pool_files():
    P = {'gbm': [os.path.join(OUT, 'v2_val_gbm.npz')], 'logreg': [os.path.join(OUT, 'v2_val_logreg.npz')]}
    for n in ('base', 'mt', 'tcn'):
        P[n] = sorted(glob.glob(os.path.join(OUT, f'v2_val_deep_{n}_s[0-9].npz')) + ([] if SMOKE else glob.glob(os.path.join(OUT, f'r3_val_{n}_s[0-9].npz'))))
    if SMOKE: return dict(P, tlob=sorted(glob.glob(os.path.join(OUT, 'lob_val_tlob_full128_s[0-9].npz'))))
    for f in glob.glob(os.path.join(OUT, 'r3_val_mt_*_s[0-9].npz')):
        n = re.match(r'r3_val_(mt_\w+?)_s\d\.npz', os.path.basename(f)).group(1); P.setdefault(n, []).append(f)
    P = {n: sorted(f) for n, f in P.items() if not n.startswith('mt_') or len(f) >= 5}
    t = sorted(glob.glob(os.path.join(OUT, 'lob_val_tlob_full128_s[0-9].npz')))
    if POOL == 'with_tlob' and len(t) == 3: P['tlob'] = t
    for f in glob.glob(os.path.join(OUT, 'r3_val_gbm_*.npz')): P[os.path.basename(f)[7:-4]] = [f]
    return P


def load_pool(P):
    Z = {n: [np.load(f) for f in fs] for n, fs in P.items()}
    keys = None
    for zs in Z.values():
        for z in zs: k = KEY(z); keys = k if keys is None else np.intersect1d(keys, k)
    out = {}
    for n, zs in Z.items():
        pr = []
        for z in zs:
            k = KEY(z); ix = np.searchsorted(k, keys); assert (k[ix] == keys).all(); pr.append(z['prob'][ix].astype(np.float64))
            if 'y' not in out: out['y'] = z['y'][ix].astype(int)
            else: assert (z['y'][ix] == out['y']).all()
        out[n] = np.mean(pr, 0)
    return out, keys


def r2_winner(Pr, keys):
    """the round-2 winner exactly as chosen in ens_v2.py: 5-seed bags (seeds 0-4) of mt and tcn, the gbm, m = 1.1"""
    pr = [Pr['gbm']]
    for n in ('mt', 'tcn'):
        q = []
        for s in range(5):
            z = np.load(os.path.join(OUT, f'v2_val_deep_{n}_s{s}.npz')); k = KEY(z); ix = np.searchsorted(k, keys); q.append(z['prob'][ix])
        pr.append(np.mean(q, 0))
    return np.mean(pr, 0)


def apply(pr, mm): return (pr * mm).argmax(1)


def best_flat(pr, y, rows=None):
    if rows is not None: pr, y = pr[rows], y[rows]
    s = {m: mf1(y, apply(pr, [1, m, 1])) for m in MS}; m = max(s, key=s.get); return s[m], [1.0, m, 1.0]


def best_cls(pr, y, rows=None, grid=MF):
    if rows is not None: pr, y = pr[rows], y[rows]
    best = (-1, None)
    for m in grid:
        for u in UD:
            f = mf1(y, apply(pr, [1, m, u]))
            if f > best[0]: best = (f, [1.0, float(m), float(u)])
    return best


def smooth(pr, keys, a):
    """causal EMA along runs of consecutive rows (a gap in the row index = a new run)"""
    out = pr.copy(); new = np.ones(len(keys), bool); new[1:] = keys[1:] != keys[:-1] + 1
    for i in np.where(~new)[0]: out[i] = (1 - a) * pr[i] + a * out[i - 1]
    return out


def context(keys):
    """time of day, spread (ticks) and short-term realised volatility of each VAL row, from the feature files"""
    allnames = [str(n) for n in E.NAMES]
    want = [allnames.index(n) for n in ('min_since_open', 'spread_ticks', 'rv_10s', 'rv_300s')]
    out = np.zeros((len(keys), len(want)), np.float32); day = keys // 1_000_000; row = keys % 1_000_000
    for d in np.unique(day):
        x, _, r, _ = E.load_day(str(d)); m = day == d; ix = np.searchsorted(r, row[m]); out[m] = x[ix][:, want]
    return out, [allnames[i] for i in want]


def stack_model(kind, cw):
    if kind == 'lr': return LogisticRegression(max_iter=300, class_weight=cw)
    return HGB(max_iter=150, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=500, class_weight=cw, random_state=0)


def stack_cv(feats, y, d, kind, cw='balanced'):
    """leave-one-day-out probabilities of a stacking model"""
    pr = np.zeros((len(y), 3))
    for k in np.unique(d):
        tr, te = d != k, d == k
        if kind == 'lr':
            mu, sd = feats[tr].mean(0), feats[tr].std(0) + 1e-6
            m = stack_model(kind, cw).fit(((feats[tr] - mu) / sd)[::3], y[tr][::3]); pr[te] = m.predict_proba((feats[te] - mu) / sd)
        else:
            m = stack_model(kind, cw)
            pr[te] = m.fit(feats[tr][::3], y[tr][::3]).predict_proba(feats[te])
    return pr


def select(Pr, y, keys, d, rows, names, ctx):
    """the whole selection on `rows`; returns the list of candidates (score, name, fn) with fn(rows_eval) -> predictions"""
    C_ = []
    for k in range(1, min(len(names), KMAX) + 1):
        for sub in combinations(names, k):
            pr = np.mean([Pr[n] for n in sub], 0); f, mm = best_flat(pr, y, rows); C_.append((f, 'avg:' + '+'.join(sub), sub, ('flat', mm)))
    C_.sort(key=lambda c: -c[0]); top = C_[:5]; extra = []
    for f0, nm, sub, _ in top:
        pr = np.mean([Pr[n] for n in sub], 0)
        f, mm = best_cls(pr, y, rows); extra.append((f, 'cls:' + nm[4:], sub, ('cls', mm)))
        for a in AL:
            ps = smooth(pr, keys, a); f, mm = best_flat(ps, y, rows); extra.append((f, f'smooth{a}:' + nm[4:], sub, ('smooth', a, mm)))
    return sorted(C_ + extra, key=lambda c: -c[0])


def predict(c, Pr, keys):
    sub, how = c[2], c[3]; pr = np.mean([Pr[n] for n in sub], 0)
    if how[0] == 'smooth': pr = smooth(pr, keys, how[1]); return apply(pr, how[2])
    return apply(pr, how[1])


if __name__ == '__main__':
    t0 = time.time(); P = pool_files(); Pr, keys = load_pool(P); y = Pr.pop('y'); d = keys // 1_000_000; names = sorted(Pr)
    log(dict(stage='info', n_rows=int(len(y)), n_days=int(len(np.unique(d))), n_files={n: len(f) for n, f in P.items()}))
    ref = r2_winner(Pr, keys); log(dict(stage='r2_winner', mF1=round(mf1(y, apply(ref, [1, 1.1, 1])), 4)))
    for n in names: f, mm = best_flat(Pr[n], y); log(dict(stage='member', name=n, n_seeds=len(P[n]), mF1_m1=round(mf1(y, Pr[n].argmax(1)), 4), mF1=round(f, 4), m=mm[1]))
    cand = select(Pr, y, keys, d, None, names, None)
    for c in cand[:25]: log(dict(stage='cand', name=c[1], mF1=round(c[0], 4), how=str(c[3])))
    # stacking (leave-one-day-out)
    ctx, cn = context(keys); lp = np.log(np.clip(np.hstack([Pr[n] for n in names]), 1e-6, 1))
    for kind in ('lr', 'gbm'):
        for use_ctx in (False, True):
            for cw in ('balanced', None):
                X = np.hstack([lp, ctx]) if use_ctx else lp; pr = stack_cv(X, y, d, kind, cw); f, mm = best_cls(pr, y, grid=MFW)
                nm = f'stack_{kind}' + ('_ctx' if use_ctx else '') + ('' if cw else '_uw') + ('' if POOL == 'main' else '_' + POOL)
                if not SMOKE: np.savez_compressed(os.path.join(OUT, f'r3_val_{nm}.npz'), prob=pr.astype(np.float32), y=y, day=d, row=keys % 1_000_000)
                log(dict(stage='stack', name=nm, ctx=cn if use_ctx else [], class_weight=cw, mF1_m1=round(mf1(y, pr.argmax(1)), 4), mF1=round(f, 4), mult=mm,
                         sec=round(time.time() - t0)))
                cand.append((f, nm, None, ('stack', kind, use_ctx, cw, mm)))
    cand.sort(key=lambda c: -c[0]); w = cand[0]
    how = w[3]; spec_ = dict(method=how[0], members=list(w[2]) if w[2] else names, mult=list(how[-1]))
    if how[0] == 'smooth': spec_['alpha'] = how[1]
    if how[0] == 'stack': spec_.update(stack_kind=how[1], ctx=bool(how[2]), class_weight=how[3])
    log(dict(stage='winner', name=w[1], mF1=round(w[0], 4), **spec_, r2_winner_mF1=round(mf1(y, apply(ref, [1, 1.1, 1])), 4)))
    # split-half check (avg / cls / smooth candidates; stacking is already scored out of sample)
    days = np.unique(d); halves = (days[:len(days) // 2], days[len(days) // 2:])
    for a, b in ((0, 1), (1, 0)):
        sel, ev = np.isin(d, halves[a]), np.isin(d, halves[b]); c = select(Pr, y, keys, d, np.where(sel)[0], names, None)[0]
        p = predict(c, Pr, keys)
        log(dict(stage='half', chosen_on=[int(x) for x in halves[a]], chosen=c[1], how=str(c[3]), mF1_other_half=round(mf1(y[ev], p[ev]), 4),
                 r2_winner_other_half=round(mf1(y[ev], apply(ref[ev], [1, 1.1, 1])), 4)))
    log(dict(stage='done', sec=round(time.time() - t0)))
