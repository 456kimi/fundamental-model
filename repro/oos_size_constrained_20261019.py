# -*- coding: utf-8 -*-
"""按裁决 d11 试做「组合构建层规模约束」：在分档内选股。

变体：
  baseline   全池 top10%（现状）
  size3      市值三分档内，各档 top10%
  size5      市值五分档内，各档 top10%
  industry   申万行业内，各行业 top10%（对照：尾部诊断显示行业亦贡献 −28bp/月）

缓冲带沿用 建仓10% / 持有25%（在档内计算）。单边成本 20bp。
只出效果，不下判死结论。
"""
import io
import os
import sys
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

REPRO = r'h:\Claw_backup\财报分析模型\repro'
OUT = os.path.join(REPRO, 'out_v2')
FAC = os.path.join(REPRO, 'factors')
COST = 0.0020
ENTRY, HOLD = 0.10, 0.25
OOS_START, OOS_END = pd.Period('2025-06', 'M'), pd.Period('2026-08', 'M')
FCOL = 'f_dir_fnn'


def log(m=''):
    print(m, flush=True)


def load():
    isf = pd.read_parquet(os.path.join(FAC, 'funda_ml_factor_panel.parquet'),
                          columns=['ts_code', 'month_end', FCOL, 'ret_fwd', 'V', 'sw_industry'])
    isf['month'] = pd.to_datetime(isf['month_end']).dt.to_period('M')
    isf['src'] = 'IS'
    o = pd.read_parquet(os.path.join(OUT, 'oos_factor_panel.parquet'),
                        columns=['ts_code', 'month', FCOL, 'ret_fwd', 'V'])
    ind = pd.read_parquet(os.path.join(REPRO, 'data_oos', 'monthly_panel_oos.parquet'),
                          columns=['ts_code', 'month_end', 'sw_industry'])
    ind['month'] = pd.to_datetime(ind['month_end']).dt.to_period('M')
    ind = ind.drop_duplicates(['ts_code', 'month'])[['ts_code', 'month', 'sw_industry']]
    o = o.merge(ind, on=['ts_code', 'month'], how='left')
    o['src'] = 'OOS'
    d = pd.concat([isf, o], ignore_index=True)
    d = d[d['V'].notna() & (d['V'] > 0)].copy()
    d['lV'] = np.log(d['V'])
    d = d.sort_values(['ts_code', 'month'])
    d[FCOL + '_s3'] = d.groupby('ts_code')[FCOL].transform(
        lambda s: s.rolling(3, min_periods=1).mean())
    return d


def bucket_series(g, strat, k):
    if strat == 'size':
        return pd.qcut(g['V'].rank(method='first'), k, labels=False)
    if strat == 'industry':
        return g['sw_industry'].fillna('未知').astype(str)
    return pd.Series('all', index=g.index)


def holdings(g_all, col, strat, k):
    prev, out = None, {}
    for mt in sorted(g_all['month'].unique()):
        g = g_all[g_all['month'] == mt].dropna(subset=[col, 'ret_fwd'])
        if len(g) < 100:
            continue
        g = g.copy()
        g['_b'] = bucket_series(g, strat, k)
        hold = set()
        for b, gb in g.groupby('_b'):
            gb = gb.sort_values(col, ascending=False)
            n_long = max(int(round(len(gb) * ENTRY)), 1)
            n_hold = max(int(round(len(gb) * HOLD)), n_long)
            order = list(gb['ts_code'])
            pool = set(order[:n_hold])
            h = set(order[:n_long])
            if prev is not None:
                keep = (prev & pool)
                add = [c for c in order if c not in keep][:max(n_long - len(keep), 0)]
                h = set(list(keep) + add)
            hold |= h
        out[mt] = hold
        prev = hold
    return out


def evaluate(d, sub, strat, k, idx):
    hs = holdings(sub, FCOL + '_s3', strat, k)
    rows = []
    prev = None
    for mt in sorted(sub['month'].unique()):
        g = sub[sub['month'] == mt].dropna(subset=[FCOL + '_s3', 'ret_fwd'])
        h = hs.get(mt)
        if not h or len(g) < 100:
            continue
        ish = g['ts_code'].isin(h)
        if ish.sum() < 20:
            continue
        to = 0.0 if prev is None else 1 - len(prev & h) / max(len(prev), 1)
        w_pf = g.loc[ish, 'sw_industry'].fillna('未知').value_counts(normalize=True)
        w_bm = g['sw_industry'].fillna('未知').value_counts(normalize=True)
        allind = w_pf.index.union(w_bm.index)
        l1 = 0.5 * (w_pf.reindex(allind, fill_value=0) - w_bm.reindex(allind, fill_value=0)).abs().sum()
        rows.append({'month': mt,
                     'long': g.loc[ish, 'ret_fwd'].mean(), 'ew': g['ret_fwd'].mean(),
                     'idx': idx.get(mt, np.nan), 'to': to,
                     'lnV_gap': g.loc[ish, 'lV'].mean() - g['lV'].mean(),
                     'ind_l1': l1, 'n': int(ish.sum())})
        prev = h
    r = pd.DataFrame(rows)
    r['ex'] = r['long'] - r['ew']
    r['net'] = r['ex'] - r['to'] * 2 * COST
    return r


def summarize(tag, r):
    ex = r['ex']
    t = ex.mean() / (ex.std(ddof=1) / np.sqrt(len(ex)))
    return {'变体': tag, '月数': len(r),
            '月均%': ex.mean() * 100, '中位%': ex.median() * 100,
            '胜率%': 100 * (ex > 0).mean(), 't': t,
            '最差月%': ex.min() * 100,
            '累计超额%': ((1 + ex).prod() - 1) * 100,
            '累计净%': ((1 + r['net']).prod() - 1) * 100,
            '换手%/月': r['to'].mean() * 100,
            'lnV偏差': r['lnV_gap'].mean(),
            '行业L1': r['ind_l1'].mean(),
            '持仓数': r['n'].mean()}


def main():
    d = load()
    idx = pd.read_parquet(r'h:\Claw_backup\panel620\benchmarks\index_daily.parquet')
    idx['trade_date'] = pd.to_datetime(idx['trade_date'])
    idx = idx[idx['ts_code'] == '000852.SH'].copy()
    idx['month'] = idx['trade_date'].dt.to_period('M')
    idx = idx.groupby('month').tail(1)
    idx['ret_idx'] = idx['close'] / idx['close'].shift(1) - 1
    idxd = idx.set_index('month')['ret_idx'].to_dict()

    periods = [('样本内 2016-25', d[(d['src'] == 'IS') & (d['month'] >= pd.Period('2016-01', 'M'))]),
               ('样本外 2025-06~2026-08',
                d[(d['src'] == 'OOS') & (d['month'] >= OOS_START) & (d['month'] <= OOS_END)])]
    variants = [('基线（全池）', 'none', 1), ('规模3档', 'size', 3), ('规模5档', 'size', 5),
                ('行业分档（对照）', 'industry', 0)]

    allsum = []
    for pname, sub in periods:
        log('\n' + '=' * 112)
        log('### %s  （%d 个月）' % (pname, sub['month'].nunique()))
        log('=' * 112)
        log('%-16s %8s %8s %6s %6s %8s %10s %10s %7s %7s %6s' % (
            '变体', '月均%', '中位%', '胜率%', 't', '最差月%', '累计超额%', '累计净%', '换手%', 'lnV偏差', '行业L1'))
        for tag, strat, k in variants:
            r = evaluate(d, sub, strat, k, idxd)
            if r.empty:
                continue
            s = summarize(tag, r)
            s['区间'] = pname
            allsum.append(s)
            log('%-16s %+8.3f %+8.3f %6.0f %+6.2f %+8.2f %+10.2f %+10.2f %7.1f %+7.2f %6.3f' % (
                s['变体'], s['月均%'], s['中位%'], s['胜率%'], s['t'], s['最差月%'],
                s['累计超额%'], s['累计净%'], s['换手%/月'], s['lnV偏差'], s['行业L1']))

    out = pd.DataFrame(allsum)
    out.to_csv(os.path.join(FAC, 'size_constrained_grid.csv'), index=False, encoding='utf-8-sig')
    log('\n已写 factors/size_constrained_grid.csv')


if __name__ == '__main__':
    main()
