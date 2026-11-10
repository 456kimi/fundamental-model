# -*- coding: utf-8 -*-
"""因子相关性矩阵 + 换手率与交易成本侵蚀评估。

输入 : factors/funda_ml_factor_panel.parquet
输出 : factors/factor_correlation.csv
       factors/turnover_cost.csv
       out_v2/corr_turnover_report.md
"""
import io, sys, os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

FACTOR_DIR = r'h:\Claw_backup\财报分析模型\repro\factors'
OUT_DIR = r'h:\Claw_backup\财报分析模型\repro\out_v2'
PQ = os.path.join(FACTOR_DIR, 'funda_ml_factor_panel.parquet')

FACTORS = ['f_mis_xgb', 'f_mis_xgb_sn', 'f_mis_fnn', 'f_mis_fnn_sn', 'f_dir_gbdt', 'f_dir_fnn']
LABEL = {
    'f_mis_xgb': '错配XGB', 'f_mis_xgb_sn': '错配XGB(中性)', 'f_mis_fnn': '错配FNN',
    'f_mis_fnn_sn': '错配FNN(中性)', 'f_dir_gbdt': '直接GBDT', 'f_dir_fnn': '直接FNN',
}
# A股单边成本假设（佣金+印花税+滑点）
COSTS = [0.0010, 0.0015, 0.0020, 0.0030]

df = pd.read_parquet(PQ)
df['month'] = pd.to_datetime(df['month_end']).dt.to_period('M')
df['year'] = df['month'].astype(str).str[:4].astype(int)
df = df.sort_values(['ts_code', 'month'])
print('面板 %d 行 | %d 月 | %d 股' % (len(df), df['month'].nunique(), df['ts_code'].nunique()))


# ------------------------------------------------------------ 1) 相关性矩阵
def avg_rank_corr(sub, a, b):
    """逐月横截面 Spearman 相关的时间序列均值。"""
    cs = []
    for mt, g in sub.dropna(subset=[a, b]).groupby('month'):
        if len(g) < 50:
            continue
        c, _ = spearmanr(g[a], g[b])
        if not np.isnan(c):
            cs.append(c)
    cs = pd.Series(cs)
    return (cs.mean(), len(cs)) if len(cs) else (np.nan, 0)


def corr_matrix(sub):
    m = pd.DataFrame(index=FACTORS, columns=FACTORS, dtype=float)
    for i, a in enumerate(FACTORS):
        for j, b in enumerate(FACTORS):
            if j < i:
                m.loc[a, b] = m.loc[b, a]
            elif i == j:
                m.loc[a, b] = 1.0
            else:
                v, _ = avg_rank_corr(sub, a, b)
                m.loc[a, b] = v
    return m


print()
print('=' * 90)
print('【1】因子相关性矩阵（逐月横截面 Spearman 均值）')
print('=' * 90)
for tag, sub in [('全期', df), ('2016-25', df[df.year >= 2016])]:
    cm = corr_matrix(sub)
    cm.index = [LABEL[c] for c in cm.index]
    cm.columns = [LABEL[c] for c in cm.columns]
    print()
    print('--- %s ---' % tag)
    print(cm.round(3).to_string())
    if tag == '2016-25':
        cm.to_csv(os.path.join(FACTOR_DIR, 'factor_correlation.csv'), encoding='utf-8-sig')


# ------------------------------------------------------------ 2) 换手率 + 成本
def decile_members(sub, col, k=10, side='long'):
    """每月末组合成员集合。"""
    memb = {}
    for mt, g in sub.dropna(subset=[col]).groupby('month'):
        if len(g) < 50:
            continue
        g = g.copy()
        try:
            g['d'] = pd.qcut(g[col].rank(method='first'), k, labels=False) + 1
        except Exception:
            continue
        tgt = k if side == 'long' else 1
        memb[mt] = set(g.loc[g['d'] == tgt, 'ts_code'])
    return memb


def turnover(memb):
    """月度单边换手率 = 本期新进名称占比。"""
    ms = sorted(memb)
    tos = []
    for i in range(1, len(ms)):
        prev, cur = memb[ms[i - 1]], memb[ms[i]]
        if not prev or not cur:
            continue
        tos.append(1 - len(prev & cur) / len(prev))
    s = pd.Series(tos)
    return s.mean() if len(s) else np.nan


def net_hl(sub, col, cost, k=10):
    """扣除成本后的多空收益（月度 %）。成本按 |Δw| 计提。"""
    rows = []
    prev_l = prev_s = None
    for mt, g in sub.dropna(subset=[col, 'ret_fwd']).groupby('month'):
        if len(g) < 50:
            continue
        g = g.copy()
        try:
            g['d'] = pd.qcut(g[col].rank(method='first'), k, labels=False) + 1
        except Exception:
            continue
        long_m = set(g.loc[g['d'] == k, 'ts_code'])
        short_m = set(g.loc[g['d'] == 1, 'ts_code'])
        rl = g.loc[g['d'] == k, 'ret_fwd'].mean()
        rs = g.loc[g['d'] == 1, 'ret_fwd'].mean()
        gross = rl - rs
        if prev_l is None:
            c = 0.0
        else:
            to_l = 1 - len(prev_l & long_m) / max(len(prev_l), 1)
            to_s = 1 - len(prev_s & short_m) / max(len(prev_s), 1)
            c = (to_l + to_s) * cost * 2      # 单边换手 × 2(卖+买) × 单边成本
        rows.append(gross - c)
        prev_l, prev_s = long_m, short_m
    s = pd.Series(rows).dropna()
    if len(s) < 2 or s.std() == 0:
        return np.nan, np.nan
    return s.mean() * 100, s.mean() / (s.std() / np.sqrt(len(s)))


print()
print('=' * 110)
print('【2】换手率与成本侵蚀（等权十分位多空）')
print('=' * 110)
rows = []
for tag, sub in [('全期', df), ('2016-25', df[df.year >= 2016])]:
    for c in FACTORS:
        to_l = turnover(decile_members(sub, c, 10, 'long'))
        to_s = turnover(decile_members(sub, c, 10, 'short'))
        g0, t0 = net_hl(sub, c, 0.0)
        rec = {'区间': tag, '因子': LABEL[c],
               '多头换手%': round(to_l * 100, 1), '空头换手%': round(to_s * 100, 1),
               '综合换手%': round((to_l + to_s) / 2 * 100, 1),
               '年化换手%': round((to_l + to_s) / 2 * 12 * 100, 0),
               '毛收益%': round(g0, 3), '毛t': round(t0, 2)}
        for cost in COSTS:
            n, tn = net_hl(sub, c, cost)
            rec['净_%dbp' % (cost * 10000)] = round(n, 3)
        rows.append(rec)

tt = pd.DataFrame(rows)
tt.to_csv(os.path.join(FACTOR_DIR, 'turnover_cost.csv'), index=False, encoding='utf-8-sig')
print()
print(tt.to_string(index=False))

# ------------------------------------------------------------ 3) 报告
lines = ['# 因子相关性与换手成本评估', '',
         '样本：%d 行 / %d 月 / %d 股' % (len(df), df['month'].nunique(), df['ts_code'].nunique()), '']

cm = corr_matrix(df[df.year >= 2016])
cm.index = [LABEL[c] for c in cm.index]
cm.columns = [LABEL[c] for c in cm.columns]
lines += ['## 一、相关性矩阵（2016-25，逐月横截面 Spearman 均值）', '',
          '| 因子 | ' + ' | '.join(cm.columns) + ' |',
          '|' + '---|' * (len(cm.columns) + 1)]
for i in cm.index:
    lines.append(('| %s | ' + ' | '.join(['%.3f'] * len(cm.columns)) + ' |') % ((i,) + tuple(cm.loc[i])))

lines += ['', '## 二、换手率与成本侵蚀（等权多空）', '',
          '| 区间 | 因子 | 综合换手% | 年化换手% | 毛收益% | 净@10bp | 净@15bp | 净@20bp | 净@30bp |',
          '|---|---|---|---|---|---|---|---|---|']
for _, r in tt.iterrows():
    lines.append('| %s | %s | %.1f | %.0f | %.3f | %.3f | %.3f | %.3f | %.3f |' % (
        r['区间'], r['因子'], r['综合换手%'], r['年化换手%'], r['毛收益%'],
        r['净_10bp'], r['净_15bp'], r['净_20bp'], r['净_30bp']))

lines += ['', '## 三、口径说明', '',
          '- 换手率 = 月度组合中"新进名称占比"，多头/空头分别计算',
          '- 净收益 = 毛收益 − (多头换手 + 空头换手) × 2 × 单边成本',
          '  （换手一次需卖出旧仓 + 买入新仓，故 ×2）',
          '- 成本档位：10/15/20/30 bp 单边（含佣金、印花税、滑点）',
          '- 等权口径；未考虑冲击成本、涨跌停不可成交、停牌']

with open(os.path.join(OUT_DIR, 'corr_turnover_report.md'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print()
print('已写 factors/factor_correlation.csv, factors/turnover_cost.csv, out_v2/corr_turnover_report.md')
