"""Summary tables (CSV) built only from the result files in results/. No number is typed by hand.
usage: python make_tables.py -> report/tables/*.csv
"""
import _paths  # noqa: F401  (src/common, src/direction, src/volatility, src/report on the import path)
import csv, json, os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R = os.path.join(ROOT, 'results'); OUT = os.path.join(ROOT, 'report', 'tables'); os.makedirs(OUT, exist_ok=True)
def rp(f):
    """path of a results file: results/direction or results/volatility"""
    for sub in ('direction', 'volatility'):
        p = os.path.join(R, sub, f)
        if os.path.exists(p): return p
    return os.path.join(R, 'direction', f)


rd = lambda f: [json.loads(l) for l in open(rp(f))]


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

# 10. MLPLOB / TLOB vs conv+GRU (lob_models.py, LOB_MODELS.md)
lob = rd('lob_models.jsonl'); base = [r for r in rd('v2_deep.jsonl') if r.get('name') == 'base' and r.get('seed') is not None]
inp = {'raw': '40 raw book columns', 'full': '127 features'}
rows = [['conv+GRU', inp['raw'], 50, 40, 0, rc['gru_curve']['n_params'], '', '', '', '', rc['val']['gru_raw']['mF1'], rc['val']['gru_raw']['acc'], '']]
rows += [['conv+GRU', inp['full'], 50, r['n_features'], r['seed'], r['n_params'], 0.001, '', r['best_epoch'] + 1, r['best_weights'], r['val_mF1'], r['val_acc'],
          round(r['sec'] / 60, 1)] for r in base]
rows += [[r['arch'].upper(), inp[r['cols']], r['T'], r['n_features'], r['seed'], r['n_params'], r['lr'], r['layers'], r['best_epoch'] + 1, r['best_weights'],
          r['val_mF1'], r['val_acc'], r['curve'][-1]['min']] for r in lob if r.get('stage') in ('S', 'M')]
write('12_lob_models_val.csv', ['model', 'input', 'snapshots', 'n_features', 'seed', 'n_params', 'learning_rate', 'layers', 'best_epoch (1 = first)', 'best_weights',
                                'val_macro_F1', 'val_accuracy', 'train_minutes'], rows)
bv = [r for r in lob if r.get('stage') == 'bagval'][-1]; te = [r for r in lob if r.get('stage') == 'test'][-1]
vs = {r['seed']: r['val_mF1'] for r in base}
rows = [['conv+GRU (127 features, 50 snapshots)', f'seed {s}', vs[s], t] for s, t in zip(te['seeds'], te['convgru_seed_mF1'])]
rows += [['conv+GRU (127 features, 50 snapshots)', 'mean of 3 seeds', bv['convgru_bag_mF1'], te['convgru_bag_mF1']]]
rows += [['TLOB (127 features, 128 snapshots)', f'seed {s}', v, t] for s, v, t in zip(te['seeds'], bv['seed_mF1'], te['seed_mF1'])]
rows += [['TLOB (127 features, 128 snapshots)', 'mean of 3 seeds', bv['bag_mF1'], te['mF1']], ['conv+GRU and TLOB averaged', 'mean of 6', bv['both_mF1'], te['both_mF1']]]
b = te['boot_vs_convgru']
rows += [['TLOB - conv+GRU (means of 3 seeds), day-block bootstrap', 'mean / 95% CI', '', f"{b['mean']} [{b['ci95'][0]}, {b['ci95'][1]}]"]]
write('13_lob_models_val_test.csv', ['model', 'seeds', f"val_macro_F1 (single seeds: own VAL rows; means: the same {bv['n']} rows)", f"test_macro_F1 (same {te['n']} rows)"], rows)
cost = rd('lob_cost.jsonl')
write('14_lob_models_cost.csv', ['model', 'snapshots', 'n_params', 'multiply_adds_per_window', 'x_conv+GRU', 'ms_per_256_windows_2_threads', 'ms_per_window_2_threads',
                                 'train_minutes_seed0'],
      [[r['name'].split('_')[0].replace('convgru', 'conv+GRU').upper().replace('CONV+GRU', 'conv+GRU'), r['T'], r['n_params'], r['macs'], r['macs_vs_convgru'],
        r['ms_per_256'], r['ms_per_1'], r['train_min']] for r in cost])

# 11. both tasks side by side: direction (5 s) and volatility (60 s), VAL and TEST macro-F1 of every method run on them
s5 = {r['model']: r['mF1'] for r in rd('select.jsonl') if r.get('exp') == 'horizon' and r.get('H') == 5}
gbm_val = max(r['mF1'] for r in rd('select.jsonl') if r.get('exp') == 'gbm_grid')   # the tuned GBM that final_eval.py scored on TEST
v2f = {r['variant']: r for r in rd('v2_final.jsonl') if r.get('stage') == 'test'}
r3f = rd('r3_final.jsonl')[-1]; vv2 = {r['name']: r for r in rd('vol_v2.jsonl') if r.get('stage') == 'test'}
vwin = [r for r in rd('vol_v2.jsonl') if r.get('stage') == 'E_winner'][-1]; vr3 = [r for r in rd('vol_r3.jsonl') if r.get('stage') == 'test'][-1]
na = ''
rows = [['simple rule', 'always flat / time of day only', s5['majority'], f5['majority']['mF1'], vc['base_val']['time_of_day']['mF1'], vc['test_tab']['time_of_day']['mF1']],
        ['simple rule', 'momentum / previous 60 s class', s5['momentum(sign ret20)'], f5['momentum']['mF1'], vc['base_val']['persistence']['mF1'], vc['test_tab']['persistence']['mF1']],
        ['round 1', 'logistic regression', s5['logreg all'], f5['logreg']['mF1'], vc['base_val']['logistic']['mF1'], vc['test_tab']['logistic']['mF1']],
        ['round 1', 'GBM (direction: tuned on VAL)', gbm_val, f5['gbm']['mF1'], vc['base_val']['gbm']['mF1'], vc['test_tab']['gbm']['mF1']],
        ['round 1', 'DeepLOB', deep['deeplob']['best_val_mF1'], deep['deeplob']['test_mF1'], na, na],
        ['round 1', f"conv+GRU (direction: tuned, {tg['chosen']})", tg['best_val_mF1'], tg['test_mF1'], vc['gru_val']['best_val_mF1'], vc['test_gru']['mF1']],
        ['round 2', 'best on VAL (direction: GBM + mt + tcn mean; volatility: GBM regression + thresholds)', v2f['primary']['val_mF1'], v2f['primary']['mF1'],
         vwin['val_mF1'], vv2['train']['mF1']],
        ['round 2', 'same, re-trained on TRAIN + VAL', na, v2f['secondary']['mF1'], na, vv2['train+val']['mF1']],
        ['MLPLOB / TLOB', 'TLOB, 128 snapshots, 3 seeds (direction only)', bv['bag_mF1'], te['mF1'], na, na],
        ['MLPLOB / TLOB', 'conv+GRU reference, 3 seeds (same rows)', bv['convgru_bag_mF1'], te['convgru_bag_mF1'], na, na],
        ['round 3', 'stacking (best on VAL)', r3f['val_mF1'], r3f['mF1'], vr3['val_mF1'], vr3['mF1']]]
vr4 = {r['stage']: r for r in rd('vol_r4.jsonl')} if os.path.exists(rp('vol_r4.jsonl')) else {}
if 'test' in vr4 and not vr4['test'].get('skipped'):
    rows.append(['round 4', f"volatility only: GBM regression x {vr4['test']['w']} + conv+GRU regression x {round(1 - vr4['test']['w'], 2)}", na, na,
                 vr4['winner']['mF1'], vr4['test']['mF1']])
vr5 = {r['stage']: r for r in rd('r5_combine.jsonl')} if os.path.exists(rp('r5_combine.jsonl')) else {}
if 'vol_test' in vr5 and not vr5['vol_test'].get('skipped'):
    sp = vr5['vol_test']['spec']
    rows.append(['round 5', f"volatility only: GBM regression x {sp['w']} + conv+GRU regression ({sp['src']}) x {round(1 - sp['w'], 2)}", na, na,
                 vr5['winner']['mF1'], vr5['vol_test']['mF1']])
vr7 = {r['stage']: r for r in rd('mkt_r7.jsonl')} if os.path.exists(rp('mkt_r7.jsonl')) else {}
if 'vol_test' in vr7 and not vr7['vol_test'].get('skipped'):
    sp = vr7['vol_test']['spec']
    rows.append(['round 7', f"volatility only: GBM regression with market-wide features x {sp['w']} + conv+GRU regression x {round(1 - sp['w'], 2)}",
                 na, na, vr7['vol_winner']['mF1'], vr7['vol_test']['mF1']])
write('15_two_tasks_summary.csv', ['round', 'method', 'direction_5s_val_mF1', 'direction_5s_test_mF1', 'volatility_60s_val_mF1', 'volatility_60s_test_mF1'], rows)

# 12. CPU run vs Apple GPU (MPS) run of the MLPLOB / TLOB comparison (cpu_gpu.py, LOB_MODELS.md section 9)
cg = rd('cpu_gpu.jsonl'); st = lambda s: [r for r in cg if r['stage'] == s]
rows = [[r['arch'].upper(), inp[r['cols']], r['T'], r['seed'], 'mps', r['cpu_val_mF1'], r['gpu_val_mF1'], r['gpu_minus_cpu'], r['cpu_val_acc'], r['gpu_val_acc'],
         r['cpu_best_epoch'] + 1, r['gpu_best_epoch'] + 1, r['cpu_epochs'], r['gpu_epochs'], r['curve_max_abs_diff'], r['cpu_min_per_epoch'], r['gpu_min_per_epoch']]
        for r in st('records:run')]
rows += [['conv+GRU', inp['full'], 50, r['seed'], 'cpu', r['cpu_run_val_mF1'], r['gpu_run_val_mF1'], round(r['gpu_run_val_mF1'] - r['cpu_run_val_mF1'], 4),
          r['cpu_run_val_acc'], r['gpu_run_val_acc'], r['cpu_run_best_epoch'] + 1, r['gpu_run_best_epoch'] + 1, r['epochs'][0], r['epochs'][1], r['curve_max_abs_diff'], '', '']
         for r in st('records:convgru')]
write('16_cpu_gpu_runs.csv', ['model', 'input', 'snapshots', 'seed', 'device_in_gpu_run', 'cpu_run_val_macro_F1', 'gpu_run_val_macro_F1', 'gpu_minus_cpu',
                              'cpu_run_val_accuracy', 'gpu_run_val_accuracy', 'cpu_run_best_epoch (1 = first)', 'gpu_run_best_epoch (1 = first)', 'cpu_run_epochs', 'gpu_run_epochs',
                              'max_abs_diff_of_epoch_curves', 'cpu_run_min_per_epoch', 'gpu_run_min_per_epoch'], rows)
g_ch, g_bv, g_te = st('records:choice')[-1], st('records:bagval')[-1], st('records:test')[-1]
gdiff = lambda a, b: round(b - a, 4) if isinstance(a, (int, float)) and isinstance(b, (int, float)) else ''
pair = lambda split, item, a, b: [split, item, a, b, gdiff(a, b)]
rows = [pair('VAL', 'chosen config (best of the 4 full configs, seed 0)', g_ch['cpu_chosen'], g_ch['gpu_chosen'])]
rows += [pair('VAL', f'TLOB 128, seed {s}', a, b) for s, a, b in zip((0, 1, 2), g_bv['cpu_seed_mF1'], g_bv['gpu_seed_mF1'])]
rows += [pair('VAL', f'TLOB 128, mean of 3 seeds ({g_bv["n_cpu"]} rows)', g_bv['cpu_bag_mF1'], g_bv['gpu_bag_mF1']),
         pair('VAL', 'conv+GRU, mean of 3 seeds', g_bv['cpu_convgru_bag_mF1'], g_bv['gpu_convgru_bag_mF1']),
         pair('VAL', 'conv+GRU and TLOB averaged', g_bv['cpu_both_mF1'], g_bv['gpu_both_mF1'])]
rows += [pair('TEST', f'TLOB 128, seed {s}', a, b) for s, a, b in zip((0, 1, 2), g_te['cpu_seed_mF1'], g_te['gpu_seed_mF1'])]
rows += [pair('TEST', f'TLOB 128, mean of 3 seeds ({g_te["n_cpu"]} rows)', g_te['cpu_mF1'], g_te['gpu_mF1']),
         pair('TEST', 'TLOB 128, mean of 3 seeds, accuracy', g_te['cpu_acc'], g_te['gpu_acc'])]
rows += [pair('TEST', f'conv+GRU, seed {s}', a, b) for s, a, b in zip((0, 1, 2), g_te['cpu_convgru_seed_mF1'], g_te['gpu_convgru_seed_mF1'])]
rows += [pair('TEST', 'conv+GRU, mean of 3 seeds', g_te['cpu_convgru_bag_mF1'], g_te['gpu_convgru_bag_mF1']),
         pair('TEST', 'conv+GRU and TLOB averaged', g_te['cpu_both_mF1'], g_te['gpu_both_mF1'])]
bc, bg = g_te['cpu_boot_vs_convgru'], g_te['gpu_boot_vs_convgru']
rows += [pair('TEST', 'TLOB - conv+GRU, day-block bootstrap mean', bc['mean'], bg['mean']),
         ['TEST', 'TLOB - conv+GRU, day-block bootstrap 95% CI', f"[{bc['ci95'][0]}, {bc['ci95'][1]}]", f"[{bg['ci95'][0]}, {bg['ci95'][1]}]", '']]
rows += [pair('TEST', f'TLOB 128 mean, recall of {c}', g_te['cpu_recall'][c], g_te['gpu_recall'][c]) for c in ('down', 'flat', 'up')]
rows += [pair('TEST', f'TLOB 128 mean, share of {c} predictions', g_te['cpu_pred_share'][c], g_te['gpu_pred_share'][c]) for c in ('down', 'flat', 'up')]
rows += [['TEST', 'TLOB 128 mean, largest per-day macro-F1 difference (10 days)', '', '', g_te['day_max_abs_diff']]]
write('17_cpu_gpu_val_test.csv', ['split', 'item', 'cpu_run', 'gpu_run', 'gpu_minus_cpu'], rows)
write('18_cpu_gpu_same_weights.csv', ['checkpoint (GPU run)', 'trained_on', 'val_windows (every 10th)', 'max_abs_prob_diff', 'mean_abs_prob_diff', 'predictions_changed',
                                      'val_macro_F1_on_cpu', 'val_macro_F1_on_mps'],
      [[r['checkpoint'], r['trained_on'], r['n_windows'], r['max_abs_prob_diff'], r['mean_abs_prob_diff'], r['argmax_changed'], r['cpu_mF1'], r['mps_mF1']] for r in st('same')])
write('19_cpu_gpu_speed.csv', ['model', 'snapshots', 'cpu_threads', 'cpu_ms_per_train_step_256', 'mps_ms_per_train_step_256', 'train_speedup', 'cpu_ms_forward_256',
                               'mps_ms_forward_256', 'forward_speedup', 'cpu_min_per_epoch_train_only', 'mps_min_per_epoch_train_only', 'power', 'lid'],
      [[r['name'].split('_')[0].upper().replace('CONVGRU', 'conv+GRU'), r['T'], r['cpu_threads'], r['cpu_train_ms'], r['mps_train_ms'], r['train_speedup'], r['cpu_fwd256_ms'],
        r['mps_fwd256_ms'], r['fwd_speedup'], r['cpu_epoch_train_min'], r['mps_epoch_train_min'], r['power'], r['lid']] for r in st('speed')])
