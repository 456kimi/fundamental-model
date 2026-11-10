# -*- coding: utf-8 -*-
"""出图：① 样本内多头超额累计曲线 ② OOS 多头 vs 基准 ③ 因子排序散点（IS vs OOS）"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

sys.path.insert(0, r'H:\Claw_backup\tmp_run')
import funda_longonly_oos_20261019 as E

REPRO = r'h:\Claw_backup\财报分析模型\repro'
OUT = os.path.join(REPRO, 'out_v2')
COST = E.COST
OOS_START, OOS_END = pd.Period('2025-06', 'M'), pd.Period('2026-08', 'M')

# 中文字体
for f in ['C:/Windows/Fonts/msyh.ttc', 'C:/Windows/Fonts/simhei.ttf', 'C:/Windows/Fonts/simsun.ttc']:
    if os.path.exists(f):
        font_manager.fontManager.addfont(f)
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=f).get_name()
        break
plt.rcParams['axes.unicode_minus'] = False


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


def main():
    isf = pd.read_parquet(E.IS_FACTOR)
    isf['month'] = pd.to_datetime(isf['month_end']).dt.to_period('M')
    isf = isf[['ts_code', 'month', 'f_mis_xgb_sn', 'f_mis_fnn_sn',
               'f_dir_gbdt', 'f_dir_fnn', 'ret_fwd']].copy()
    isf['src'] = 'IS'
    o = pd.read_parquet(os.path.join(OUT, 'oos_factor_panel.parquet'))
    o = o[['ts_code', 'month', 'f_mis_xgb_sn', 'f_mis_fnn_sn',
           'f_dir_gbdt', 'f_dir_fnn', 'ret_fwd']].copy()
    o['src'] = 'OOS'
    fac = pd.concat([isf, o], ignore_index=True).sort_values(['ts_code', 'month'])
    for col, _ in E.FACTORS:
        fac[col + '_s3'] = fac.groupby('ts_code')[col].transform(
            lambda s: s.rolling(3, min_periods=1).mean())
    idx = E.build_index_bench().set_index('month')['ret_idx'].to_dict()

    is_sub = fac[(fac['src'] == 'IS') & (fac['month'] >= pd.Period('2016-01', 'M'))]
    oo_sub = fac[(fac['src'] == 'OOS') & (fac['month'] >= OOS_START) & (fac['month'] <= OOS_END)]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))

    # 左：样本内累计净超额
    ax = axes[0]
    for col, label in E.FACTORS:
        r = monthly(is_sub, col + '_s3', idx)
        cum = (1 + r['net']).cumprod() - 1
        ax.plot(range(len(cum)), cum * 100, lw=1.8, label=label)
    ax.axhline(0, color='#999', lw=0.8, ls='--')
    ax.set_title('样本内 2016-2025：多头相对全市场等权的累计净超额', fontsize=12)
    ax.set_xlabel('月数'); ax.set_ylabel('累计超额 %')
    ax.legend(fontsize=9, loc='upper left'); ax.grid(alpha=0.3)

    # 右：OOS 多头 vs 两种基准
    ax = axes[1]
    r = monthly(oo_sub, 'f_dir_fnn_s3', idx)
    x = range(len(r))
    ax.plot(x, ((1 + r['long']).cumprod() - 1) * 100, lw=2.0, label='直接收益-FNN 多头', color='#c0392b')
    ax.plot(x, ((1 + r['ew']).cumprod() - 1) * 100, lw=2.0, label='全市场等权', color='#2c3e50')
    ax.plot(x, ((1 + r['idx']).cumprod() - 1) * 100, lw=2.0, label='中证1000', color='#7f8c8d', ls='--')
    ax.set_xticks(list(x)[::3])
    ax.set_xticklabels([str(m)[2:] for m in r['month'][::3]], rotation=45, fontsize=8)
    ax.axhline(0, color='#999', lw=0.8, ls='--')
    ax.set_title('样本外 2025-06~2026-08：多头 vs 基准（累计）', fontsize=12)
    ax.set_ylabel('累计收益 %')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    fig.tight_layout()
    p = os.path.join(OUT, 'longonly_oos_20261019.png')
    fig.savefig(p, dpi=150)
    print('已出图 %s' % p)


if __name__ == '__main__':
    main()
