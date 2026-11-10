# -*- coding: utf-8 -*-
"""OOS 稳健性分解：负超额是「信号消失」还是「被单月暴跌主导」。"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, r'H:\Claw_backup\tmp_run')
import funda_longonly_oos_20261019 as E

REPRO = r'h:\Claw_backup\财报分析模型\repro'
OUT_DIR = os.path.join(REPRO, 'out_v2')
OOS_START, OOS_END = pd.Period('2025-06', 'M'), pd.Period('2026-08', 'M')
COST = E.COST


def build(src):
    if src == 'IS':
        d = pd.read_parquet(E.IS_FACTOR)
        d['month'] = pd.to_datetime(d['month_end']).dt.to_period('M')
        d = d[d['month'] >= pd.Period('2016-01', 'M')]
    else:
        d = pd.read_parquet(os.path.join(OUT_DIR, 'oos_factor_panel.parquet'))
    return d


def monthly(fac, col, idx_ret):
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
        rows.append({'month': mt,
                     'long': g.loc[g['ts_code'].isin(hold), 'ret_fwd'].mean(),
                     'ew': g['ret_fwd'].mean(), 'idx': idx_ret.get(mt, np.nan), 'to': to})
        prev = hold
    r = pd.DataFrame(rows)
    r['ex_ew'] = r['long'] - r['ew']
    r['net'] = r['ex_ew'] - r['to'] * 2 * COST
    return r


def ic_series(fac, col):
    rows = []
    for mt, g in fac.dropna(subset=[col, 'ret_fwd']).groupby('month'):
        if len(g) < 30:
            continue
        rows.append({'month': mt, 'ic': g[col].corr(g['ret_fwd'], method='spearman')})
    return pd.DataFrame(rows)


def main():
    # 与主表一致：IS+OOS 合并后平滑
    isf = pd.read_parquet(E.IS_FACTOR)
    isf['month'] = pd.to_datetime(isf['month_end']).dt.to_period('M')
    isf = isf[['ts_code', 'month', 'f_mis_xgb_sn', 'f_mis_fnn_sn', 'f_dir_gbdt',
               'f_dir_fnn', 'ret_fwd']].copy()
    isf['src'] = 'IS'
    o = pd.read_parquet(os.path.join(OUT_DIR, 'oos_factor_panel.parquet'))
    o = o[['ts_code', 'month', 'f_mis_xgb_sn', 'f_mis_fnn_sn', 'f_dir_gbdt',
           'f_dir_fnn', 'ret_fwd']].copy()
    o['src'] = 'OOS'
    fac = pd.concat([isf, o], ignore_index=True).sort_values(['ts_code', 'month'])
    for col, _ in E.FACTORS:
        fac[col + '_s3'] = fac.groupby('ts_code')[col].transform(
            lambda s: s.rolling(3, min_periods=1).mean())
    idx = E.build_index_bench().set_index('month')['ret_idx'].to_dict()

    oo = fac[(fac['src'] == 'OOS') & (fac['month'] >= OOS_START) & (fac['month'] <= OOS_END)]
    iss = fac[(fac['src'] == 'IS') & (fac['month'] >= pd.Period('2016-01', 'M'))]

    print('=' * 92)
    print('### OOS 稳健性分解（2025-06 ~ 2026-08，15 个月）')
    print('=' * 92)
    print('%-20s %8s %8s %8s %8s %10s %10s' % (
        '因子', '月均', '中位', '胜率%', '最差月', '剔除最差后均', '剔除后中位'))
    for col, label in E.FACTORS:
        r = monthly(oo, col + '_s3', idx)
        ex = r['ex_ew']
        worst = ex.min()
        rest = ex.drop(ex.idxmin())
        print('%-20s %+8.3f %+8.3f %8.1f %+8.3f %+10.3f %+10.3f' % (
            label, ex.mean() * 100, ex.median() * 100, 100 * (ex > 0).mean(),
            worst * 100, rest.mean() * 100, rest.median() * 100))

    print('\n### RankIC：均值 vs 中位（看是否被单月拖累）')
    print('%-20s %10s %10s %10s %10s %8s' % (
        '因子', 'IC均值', 'IC中位', 'IC>0占比', '剔除最差月', '月数'))
    for col, label in E.FACTORS:
        ic = ic_series(oo, col + '_s3')
        v = ic['ic'].dropna()
        rest = v.drop(v.idxmin())
        print('%-20s %+10.4f %+10.4f %9.1f%% %+10.4f %8d' % (
            label, v.mean(), v.median(), 100 * (v > 0).mean(), rest.mean(), len(v)))

    print('\n### 对照：样本内 2016-25 的同一组指标')
    print('%-20s %8s %8s %8s %10s %10s' % ('因子', '月均', '中位', '胜率%', 'IC均值', 'IC中位'))
    for col, label in E.FACTORS:
        r = monthly(iss, col + '_s3', idx)
        ic = ic_series(iss, col + '_s3')['ic'].dropna()
        print('%-20s %+8.3f %+8.3f %8.1f %+10.4f %+10.4f' % (
            label, r['ex_ew'].mean() * 100, r['ex_ew'].median() * 100,
            100 * (r['ex_ew'] > 0).mean(), ic.mean(), ic.median()))

    # 极端月定位
    r = monthly(oo, 'f_dir_fnn_s3', idx)
    w = r.loc[r['ex_ew'].idxmin()]
    print('\n### 最差月定位（直接收益-FNN）')
    print('  月份 %s | 多头 %.2f%% | 等权 %.2f%% | 超额 %.2f%%'
          % (w['month'], w['long'] * 100, w['ew'] * 100, w['ex_ew'] * 100))
    print('  该月对 15 个月累计超额的贡献: %.2f%%（占累计 %.2f%% 的 %.0f%%）'
          % (w['ex_ew'] * 100, r['ex_ew'].sum() * 100,
             100 * w['ex_ew'] / r['ex_ew'].sum() if r['ex_ew'].sum() != 0 else np.nan))


if __name__ == '__main__':
    main()
