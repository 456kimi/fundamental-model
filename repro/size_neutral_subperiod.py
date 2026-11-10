# -*- coding: utf-8 -*-
"""子期间稳健性：非线性模型原始 vs 规模中性化 H-L，分 2009-15 / 2016-25。"""
import io, sys, os
import numpy as np
import pandas as pd
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

REPRO = r'h:\Claw_backup\财报分析模型\repro'
IN = os.path.join(REPRO, 'out_v2', 'mismatch_panel_v2.parquet')
MODELS = ['M_DT', 'M_RF', 'M_GBDT', 'M_XGBOOST', 'M_FNN', 'M_NONL', 'M_PCR', 'M_OLS']


def decile_hl(df, c):
    hl = []
    for mt, g in df.dropna(subset=[c, 'ret_fwd']).groupby('month'):
        if len(g) < 50:
            continue
        g = g.copy()
        try:
            g['d'] = pd.qcut(g[c].rank(method='first'), 10, labels=False) + 1
        except Exception:
            continue
        d = g.groupby('d')['ret_fwd'].mean()
        if 10 in d and 1 in d:
            hl.append(d[10] - d[1])
    hl = pd.Series(hl).dropna()
    if len(hl) < 2 or hl.std() == 0:
        return np.nan, np.nan, len(hl)
    return hl.mean() * 100, hl.mean() / (hl.std() / np.sqrt(len(hl))), len(hl)


df = pd.read_parquet(IN)
df = df.dropna(subset=['ret_fwd', 'V'])
df = df[df['V'] > 0]
df['lV'] = np.log(df['V'])
df['year'] = df['month'].astype(str).str[:4].astype(int)

# 中性化
for c in MODELS:
    res = np.full(len(df), np.nan)
    ok = df[c].notna() & df['lV'].notna()
    for mt, g in df.loc[ok].groupby('month'):
        if len(g) < 20:
            continue
        X = np.column_stack([np.ones(len(g)), g['lV'].values])
        y = g[c].values
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        res[g.index.values] = y - X @ beta
    df[c + '_res'] = res

for tag, sub in [('全期', df), ('2009-15', df[df.year <= 2015]), ('2016-25', df[df.year >= 2016])]:
    print('\n===== %s (N=%d 月) =====' % (tag, sub['month'].nunique()))
    print('%-12s %16s %20s' % ('model', '原始 HL% (t)', '中性化 HL% (t)'))
    for c in MODELS:
        m0, t0, n0 = decile_hl(sub, c)
        m1, t1, n1 = decile_hl(sub, c + '_res')
        print('%-12s %7.3f (%5.2f) %11.3f (%5.2f)' % (c, m0, t0, m1, t1))
