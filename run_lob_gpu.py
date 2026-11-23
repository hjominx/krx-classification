"""One-command run of the MLPLOB / TLOB comparison on a machine with a GPU (NVIDIA CUDA on Windows / Linux, or Apple MPS on macOS).

usage:  python run_lob_gpu.py <folder with the raw .jsonl.gz files>

The raw folder may contain the order-book (H0STASP0) and trade (H0STCNT0) files in any sub-folders and under any names;
each file is classified by its first record. Every step skips work that is already done, so the command can simply be
run again after an interruption. Results go to results_gpu/direction/ (kept apart from the CPU results in results/direction/).
Steps: 1 check GPU  2 raw -> per-day npz  3 features (book, trades, v2)  4 conv+GRU reference (3 seeds)
       5 MLPLOB / TLOB: screen (6 configs), 3-seed bag of the best, VAL comparison, TEST once
"""
import glob, gzip, json, os, shutil, subprocess, sys, time, zlib

ROOT = os.path.dirname(os.path.abspath(__file__)); SRC = os.path.join(ROOT, 'src'); PRE = os.path.join(ROOT, 'preprocess')
DATA = os.path.join(ROOT, 'data'); RESULTS = os.path.join(ROOT, 'results_gpu')
ENV = dict(os.environ, KRX_RESULTS=RESULTS, PYTHONUNBUFFERED='1')


def step(title): print(f'\n=== {title}  ({time.strftime("%H:%M:%S")})', flush=True)


def py(args, cwd=os.path.join(SRC, 'common')):
    print('  $ python', ' '.join(args), flush=True)
    r = subprocess.run([sys.executable] + args, cwd=cwd, env=ENV)
    if r.returncode: sys.exit(f'step failed: python {" ".join(args)}')


def first_record(path):
    """first JSON line of a (possibly multi-member or damaged) gzip file"""
    raw = open(path, 'rb').read(4 << 20); d = zlib.decompressobj(16 + zlib.MAX_WBITS)
    try: text = d.decompress(raw)
    except zlib.error: return None
    line = text.split(b'\n', 1)[0]
    try: return json.loads(line)
    except ValueError: return None


if __name__ == '__main__':
    if len(sys.argv) < 2: sys.exit(__doc__)
    raw_dir = os.path.abspath(sys.argv[1])

    step('1. GPU check')
    import torch
    if torch.cuda.is_available(): print('  GPU:', torch.cuda.get_device_name(0))
    elif torch.backends.mps.is_available(): print('  GPU: Apple MPS (Metal)')
    else: print('  WARNING: PyTorch does not see a GPU - the run will use the CPU and be slow.\n'
                '  Install the CUDA build of PyTorch (https://pytorch.org/get-started/locally/) and run again.')

    step('2. raw files -> per-day npz')
    book_dir, trade_raw, trade_dir = (os.path.join(DATA, d) for d in ('krx_npz', 'krx_trades_raw', 'krx_trades'))
    for d in (book_dir, trade_raw, trade_dir): os.makedirs(d, exist_ok=True)
    files = sorted(glob.glob(os.path.join(raw_dir, '**', '*.jsonl.gz'), recursive=True)); done_log = os.path.join(DATA, 'converted.txt')
    done = set(open(done_log).read().split('\n')) if os.path.exists(done_log) else set()
    n_book = n_trade = 0
    for f in files:
        r = first_record(f)
        if not r: print('  skipped (unreadable first line):', f); continue
        kind = 'trade' if (r.get('tr_id') == 'H0STCNT0' or r.get('event_type') == 'trade') else 'book' if r.get('event_type') == 'orderbook' or r.get('tr_id') == 'H0STASP0' else None
        if kind == 'book':
            n_book += 1
            if f not in done: py([os.path.join(PRE, 'convert_krx.py'), f, book_dir, PRE], cwd=PRE); open(done_log, 'a').write(f + '\n')
        elif kind == 'trade':
            n_trade += 1; day = str(r.get('recv_ts', ''))[:10]
            dst = os.path.join(trade_raw, f'{day}.jsonl.gz')
            if len(day) == 10 and not os.path.exists(dst): shutil.copyfile(f, dst)
    print(f'  order-book files {n_book}, trade files {n_trade}, days with a book file {len(os.listdir(book_dir))}')
    if len(os.listdir(trade_dir)) < len(os.listdir(trade_raw)): py([os.path.join(PRE, 'convert_trades.py'), trade_raw, trade_dir, PRE], cwd=PRE)

    step('3. features')
    for sub, script in (('feat', 'build_features.py'), ('feat_tr', 'trade_features.py'), ('feat2', 'features_v2.py')):
        have = len(glob.glob(os.path.join(DATA, sub, '*.npz')))
        if have < len(os.listdir(book_dir)) - (1 if sub == 'feat_tr' else 0): py([script])
        else: print(f'  {sub}: {have} days already built')

    step('4. conv+GRU reference (same rules as the CPU run, seeds 0-2)')
    py(['deep_v2.py', 'base:0,base:1,base:2', '4'], cwd=os.path.join(SRC, 'direction'))

    step('5. MLPLOB / TLOB')
    for stage in ('S', 'M', 'bagval', 'test'): py(['lob_models.py', stage], cwd=os.path.join(SRC, 'direction'))
    print('\nresults:', os.path.join(RESULTS, 'direction', 'lob_models.jsonl'))
    for l in open(os.path.join(RESULTS, 'direction', 'lob_models.jsonl')):
        r = json.loads(l); print('  ', {k: v for k, v in r.items() if k in ('stage', 'name', 'seed', 'device', 'val_mF1', 'bag_mF1', 'convgru_bag_mF1', 'mF1', 'boot_vs_convgru')})
