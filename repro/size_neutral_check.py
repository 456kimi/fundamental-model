# -*- coding: utf-8 -*-
"""规模中性化诊断：验证论文异象是否独立于规模因子。

对 M 指标逐月做横截面回归  M_i = a + b * log(V_i) + e_i，取残差 e_i 作为中性化指标；
分别用原始 M 与中性化 M 做十分位多空，并对比结果。
"""
import io, sys, os
import numpy as np
import pandas as pd
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

REPRO = r'h:\Claw_backup\财报分析模型\repro'
IN = os.path.join(REPRO, 'out_v2', 'mismatch_panel_v2.parquet')
OUT_DIR = os.path.join(REPRO, 'out_v2')

MODELS = ['M_OLS', 'M_RIDGE', 'M_LASSO', 'M_ELASTICNET', 'M_PCR',
          'M_DT', 'M_RF', 'M_GBDT', 'M_XGBOOST', 'M_FNN', 'M_L', 'M_NONL', 'M_COM']


def load():
    df = pd.read_parquet(IN)
    df = df.dropna(subset=['ret_fwd', 'V'])
    df = df[df['V'] > 0]
    df['lV'] = np.log(df['V'])
    return df


def neutralize(df):
    """逐月横截面回归 M ~ log(V)，取残差。"""
    out = df.copy()
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
        out[c + '_res'] = res
    return out


def decile_hl(df, c):
    """十分位多空（等权）。"""
    hl = []
    dec = {}
    for mt, g in df.dropna(subset=[c, 'ret_fwd']).groupby('month'):
        if len(g) < 50:
            continue
        g = g.copy()
        try:
            g['d'] = pd.qcut(g[c].rank(method='first'), 10, labels=False) + 1
        except Exception:
            continue
        d = g.groupby('d')['ret_fwd'].mean()
        for k, v in d.items():
            dec.setdefault(k, []).append(v)
        if 10 in d and 1 in d:
            hl.append(d[10] - d[1])
    hl = pd.Series(hl).dropna()
    if len(hl) < 2 or hl.std() == 0:
        return np.nan, np.nan, len(hl), dec
    mean = hl.mean()
    t = mean / (hl.std() / np.sqrt(len(hl)))
    return mean * 100, t, len(hl), dec


def corr_to_size(df, c):
    """指标与 log(V) 的全样本相关系数。"""
    return df[[c, 'lV']].corr().iloc[0, 1]


def main():
    df = load()
    print('样本 %d 行 | %d 个月' % (len(df), df['month'].nunique()))

    df = neutralize(df)

    rows = []
    for c in MODELS:
        m0, t0, n0, _ = decile_hl(df, c)
        m1, t1, n1, dec1 = decile_hl(df, c + '_res')
        cr = corr_to_size(df, c)
        rows.append({
            'model': c,
            'raw_HL%': round(m0, 4), 'raw_t': round(t0, 3),
            'neut_HL%': round(m1, 4), 'neut_t': round(t1, 3),
            'corr(logV)': round(cr, 3),
            '衰减%': round(100 * (1 - (m1 / m0)) if m0 and not np.isnan(m0) else np.nan, 1),
        })
    r = pd.DataFrame(rows).set_index('model')
    r.to_csv(os.path.join(OUT_DIR, 'size_neutral_check.csv'), encoding='utf-8-sig')

    print()
    print('===== 规模中性化多空对比（等权 H-L 月度 %）=====')
    print(r.to_string())

    print()
    print('===== 判定 =====')
    key = r.loc['M_COM']
    print('M_COM  原始 H-L %.3f%% (t=%.2f) -> 中性化 %.3f%% (t=%.2f)' % (
        key['raw_HL%'], key['raw_t'], key['neut_HL%'], key['neut_t']))
    print('M_COM  与 log(V) 相关性: %.3f' % key['corr(logV)'])
    if abs(key['neut_t']) >= 2:
        print('→ 中性化后仍显著(t>=2)：异象独立于规模，论文方向可信')
    elif abs(key['neut_t']) >= 1:
        print('→ 中性化后边际显著(1<=t<2)：部分依赖规模')
    else:
        print('→ 中性化后不显著(t<1)：异象基本等于规模效应')


if __name__ == '__main__':
    main()
