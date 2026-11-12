"""Summary tables (CSV) built only from the result files in results/. No number is typed by hand.
usage: python make_tables.py -> report/tables/*.csv
"""
import csv, json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, 'results'); OUT = os.path.join(ROOT, 'report', 'tables'); os.makedirs(OUT, exist_ok=True)
rd = lambda f: [json.loads(l) for l in open(os.path.join(R, f))]


def write(name, header, rows):
    with open(os.path.join(OUT, name), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    print('wrote', name, len(rows), 'rows')


# 1. direction, 5 s, TEST
fin = {r['H']: r for r in rd('final.jsonl')}; f5 = fin[5]
deep = {r['model']: r for r in rd('deep_test.jsonl')}
tg = rd('tune_gru_final.jsonl')[-1]
tr = {r['name']: r for r in rd('trades.jsonl') if r.get('stage') == 'final'}
flat = [r for r in rd('gru_flat.jsonl') if r['stage'] == 'test']
full = [r for r in rd('gru_full.jsonl') if r.get('stage') == 'final'][-1]
rows = [['always flat (majority)', f5['majority']['mF1'], f5['majority']['acc']],
        ['momentum rule', f5['momentum']['mF1'], f5['momentum']['acc']],
        ['logistic regression', f5['logreg']['mF1'], f5['logreg']['acc']],
        ['GBM (book features)', f5['gbm']['mF1'], f5['gbm']['acc']],
        ['GBM (book + trade features)', tr['gbm_H5']['mF1'], tr['gbm_H5']['acc']],
        ['DeepLOB', deep['deeplob']['test_mF1'], deep['deeplob']['test_acc']],
        ['conv+GRU (first run)', deep['gru']['test_mF1'], deep['gru']['test_acc']],
        [f"conv+GRU (tuned, {tg['chosen']})", tg['test_mF1'], tg['test_acc']]]
rows += [[f"conv+GRU (tuned) + flat calibration m={r['m']}", r['mF1'], r['acc']] for r in flat if r['m'] != 1]
rows += [[f"conv+GRU (full-data training, {full['chosen']}) m={k[6:]}", v['mF1'], v['acc']] for k, v in full.items() if k.startswith('test_m')]
write('01_direction_5s_test.csv', ['model', 'test_macro_F1', 'test_accuracy'], rows)

# 2. horizon: VAL (selection) and TEST
sel = [r for r in rd('select.jsonl') if r.get('exp') == 'horizon']
write('02_horizon_val.csv', ['horizon_s', 'model', 'val_macro_F1', 'val_accuracy'], [[r['H'], r['model'], r['mF1'], r['acc']] for r in sel])
write('03_horizon_test.csv', ['horizon_s', 'majority_mF1', 'momentum_mF1', 'logreg_mF1', 'gbm_mF1', 'gbm_acc', 'gbm_with_trades_mF1'],
      [[h, r['majority']['mF1'], r['momentum']['mF1'], r['logreg']['mF1'], r['gbm']['mF1'], r['gbm']['acc'], tr.get(f'gbm_H{h}', {}).get('mF1', '')]
       for h, r in sorted(fin.items())])

# 3. GRU tuning on VAL
write('04_gru_tuning_val.csv', ['config', 'T', 'hidden', 'lr', 'dropout', 'class_weight', 'seed', 'val_macro_F1', 'val_accuracy', 'best_epoch'],
      [[r['name'], r['T'], r['hidden'], r['lr'], r['dropout'], r['cw'], r['seed'], r['best_val_mF1'], r['best_val_acc'], r['best_epoch']] for r in rd('tune_gru.jsonl')])

# 4. full-data training curves
write('05_full_training_curve.csv', ['config', 'epoch', 'train_loss', 'val_macro_F1', 'val_accuracy', 'learning_rate'],
      [[r['name'], c['epoch'] + 1, c['train_loss'], c['val_mF1'], c['val_acc'], c['lr']] for r in rd('gru_full.jsonl') if 'curve' in r for c in r['curve']])

# 5. trade features: stages A/B/C on VAL
rows = []
for r in rd('trades.jsonl'):
    if r['stage'] == 'A': rows.append(['A: add trades', r['name'], r['n_features'], r['mF1'], ''])
    elif r['stage'] == 'B' and 'group' in r: rows.append(['B: drop one group', r['group'], r['n_features'], r['mF1'], r['contribution']])
    elif r['stage'] == 'B': rows.append(['B: kept set', r['name'], r['n_features'], r['mF1'], ''])
    elif r['stage'] == 'C': rows.append(['C: GBM settings', r['name'], '', r['mF1'], f"best_iter={r['best_iter']}"])
write('06_trade_features_val.csv', ['stage', 'item', 'n_features', 'val_macro_F1', 'contribution_or_note'], rows)

# 6. raw-input comparison + bootstrap
rc = {r['stage']: r for r in rd('raw_cmp.jsonl')}
names = {'logistic_raw': 'logistic regression', 'gbm_raw': 'GBM (current snapshot)', 'gbm_raw_lags': 'GBM (7 lags)', 'gru_raw': 'conv+GRU (50 snapshots)'}
b = rc['test']['boot_gru_minus']
write('07_raw_input_comparison.csv', ['model (raw 40 book inputs)', 'val_macro_F1', 'test_macro_F1', 'test_accuracy', 'GRU_minus_model_mean', 'GRU_minus_model_ci95_low', 'GRU_minus_model_ci95_high'],
      [[v, rc['val'][k]['mF1'], rc['test'][k]['mF1'], rc['test'][k]['acc'], b.get(k, {}).get('mean', ''),
        *(b[k]['ci95'] if k in b else ['', ''])] for k, v in names.items()])

# 7. volatility classification
vc = {r['stage']: r for r in rd('vol_cls.jsonl')}
rows = [[n, vc['base_val'][k]['mF1'], vc['test_tab'][k]['mF1'], vc['test_tab'][k]['acc']] for k, n in
        (('persistence', 'previous 60 s class'), ('time_of_day', 'time of day only'), ('logistic', 'logistic regression'), ('gbm', 'GBM'))]
rows.append(['conv+GRU', vc['gru_val']['best_val_mF1'], vc['test_gru']['mF1'], vc['test_gru']['acc']])
write('08_volatility_60s.csv', ['model', 'val_macro_F1', 'test_macro_F1', 'test_accuracy'], rows)
cm = vc['test_gru']['confusion']
write('09_volatility_confusion_gru_test.csv', ['actual \\ predicted', 'calm', 'normal', 'turbulent'],
      [[lab] + row for lab, row in zip(('calm', 'normal', 'turbulent'), cm)])

# 8. selective prediction (next mid change)
ns = rd('next_move_sel.jsonl')
write('10_selective_next_move.csv', ['target_accuracy', 'threshold', 'val_coverage', 'test_coverage', 'test_accuracy', 'test_n'],
      [[r['target'], r['threshold'], r['val_cov'], r['test_cov'], r['test_acc'], r['test_n']] for r in ns if 'target' in r])

# 9. backtest (cost-aware)
bt = rd('backtest.jsonl')[0]
rows = []
for side in ('long_only', 'long_short'):
    s = bt[side]
    for sc in ('mid_no_cost', 'spread_only', 'passive_upper_bound', 'realistic'):
        rows.append([side, sc, s['trades'], s['signal_correct'], s[sc]['per_trade_krw'], s[sc]['total_krw'], s[sc]['win_rate']])
write('11_backtest_test.csv', ['side', 'cost_scenario', 'trades', 'signal_correct', 'pnl_per_trade_krw', 'pnl_total_krw', 'win_rate'], rows)
