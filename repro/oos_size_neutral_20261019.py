# -*- coding: utf-8 -*-
"""尾部诊断的延伸：既然样本外负超额来自风格漂移（规模），
那么把直接收益因子做「规模中性」后，样本外是否恢复？

f_dir_fnn_sn = 逐月横截面  f_dir_fnn ~ log(V)  的残差（与 f_mis_*_sn 同一做法）
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, r'H:\Claw_backup\tmp_run')
import funda_longonly_oos_20261019 as E

REPRO = r'h:\Claw_backup\财报分析模型\repro'
OUT = os.path.join(REPRO, 'out_v2')
COST = E.COST
OOS_START, OOS_END = pd.Period('2025-06', 'M'), pd.Period('2026-08', 'M')


def sn(df, col):
    out = np.full(len(df), np.nan)
    ok = df[col].notna() & df['V'].notna() & (df['V'] > 0)
    for mt, g in df.loc[ok].groupby('month'):
        if len(g) < 30:
            continue
        X = np.column_stack([np.ones(len(g)), np.log(g['V'].values)])
        y = g[col].values
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        out[g.index.values] = y - X @ b
    return out


def build():
    isf = pd.read_parquet(E.IS_FACTOR)
    isf['month'] = pd.to_datetime(isf['month_end']).dt.to_period('M')
    isf = isf[['ts_code', 'month', 'f_dir_fnn', 'f_dir_gbdt', 'ret_fwd', 'V']].copy()
    isf['src'] = 'IS'
    o = pd.read_parquet(os.path.join(OUT, 'oos_factor_panel.parquet'))
    o = o[['ts_code', 'month', 'f_dir_fnn', 'f_dir_gbdt', 'ret_fwd', 'V']].copy()
    o['src'] = 'OOS'
    fac = pd.concat([isf, o], ignore_index=True)
    for c in ['f_dir_fnn', 'f_dir_gbdt']:
        fac[c + '_sn'] = sn(fac, c)
    fac = fac.sort_values(['ts_code', 'month'])
    for c in ['f_dir_fnn', 'f_dir_gbdt', 'f_dir_fnn_sn', 'f_dir_gbdt_sn']:
        fac[c + '_s3'] = fac.groupby('ts_code')[c].transform(
            lambda s: s.rolling(3, min_periods=1).mean())
    return fac


def monthly(fac, col, idx):
    prev, rows = None, []
    for mt in sorted(fac['month'].unique()):
        g = fac[fac['month'] == mt].dropna(subset=[col, 'ret_fwd'])
        if len(g) < 50:
            continue
        g = g.copy()
        g['rk'] = g[col].rank(method='first')
        n_long = max(int(round(len(g) * E.ENTRY_PCT)), 5)
        n_hold = max(int(round(len(g) * E.HOLD_PCT)), n_long)
        top = list(g.sort_values('rk', ascending=False)['ts_code'])
        pool = set(top[:n_hold])
        hold, to = set(top[:n_long]), 0.0
        if prev is not None:
            keep = prev & pool
            add = [c for c in top if c not in keep][:max(n_long - len(keep), 0)]
            hold = set(list(keep) + add)
            to = 1 - len(prev & hold) / max(len(prev), 1)
        rows.append({'month': str(mt), 'long': g.loc[g['ts_code'].isin(hold), 'ret_fwd'].mean(),
                     'ew': g['ret_fwd'].mean(), 'to': to,
                     'lnV_pf': np.log(g.loc[g['ts_code'].isin(hold), 'V']).mean(),
                     'lnV_bm': np.log(g['V']).mean()})
        prev = hold
    r = pd.DataFrame(rows)
    r['ex'] = r['long'] - r['ew']
    r['net'] = r['ex'] - r['to'] * 2 * COST
    return r


def show(tag, r):
    ex = r['ex']
    t = ex.mean() / (ex.std(ddof=1) / np.sqrt(len(ex)))
    print('  %-26s 月均 %+7.3f%% | 中位 %+7.3f%% | 胜率 %4.0f%% | t %+5.2f | 累计 %+7.2f%% | lnV偏差 %+.2f'
          % (tag, ex.mean() * 100, ex.median() * 100, 100 * (ex > 0).mean(), t,
             ((1 + ex).prod() - 1) * 100, (r['lnV_pf'] - r['lnV_bm']).mean()))


def main():
    fac = build()
    idx = E.build_index_bench().set_index('month')['ret_idx'].to_dict()
    is_sub = fac[(fac['src'] == 'IS') & (fac['month'] >= pd.Period('2016-01', 'M'))]
    oo_sub = fac[(fac['src'] == 'OOS') & (fac['month'] >= OOS_START) & (fac['month'] <= OOS_END)]

    for tag, sub in [('样本内 2016-25', is_sub), ('样本外 2025-06~2026-08', oo_sub)]:
        print('\n' + '=' * 108)
        print('### ' + tag)
        print('=' * 108)
        for col, label in [('f_dir_fnn', '直接收益-FNN 原始'),
                           ('f_dir_fnn_sn', '直接收益-FNN 规模中性'),
                           ('f_dir_gbdt', '直接收益-GBDT 原始'),
                           ('f_dir_gbdt_sn', '直接收益-GBDT 规模中性')]:
            show(label, monthly(sub, col + '_s3', idx))

    # 存月度明细
    rows = []
    for tag, sub in [('样本内 2016-25', is_sub), ('样本外 2025-06~2026-08', oo_sub)]:
        for col, label in [('f_dir_fnn_s3', 'FNN原始'), ('f_dir_fnn_sn_s3', 'FNN规模中性')]:
            r = monthly(sub, col, idx)
            r['区间'] = tag
            r['口径'] = label
            rows.append(r)
    pd.concat(rows, ignore_index=True).to_csv(
        os.path.join(E.FACTOR_DIR, 'size_neutral_oos.csv'), index=False, encoding='utf-8-sig')
    print('\n已写 factors/size_neutral_oos.csv')


if __name__ == '__main__':
    main()
