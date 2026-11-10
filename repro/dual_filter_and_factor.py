# -*- coding: utf-8 -*-
"""① 双重过滤稳健性（等权/市值加权 × 原始/规模中性/规模+行业中性 × 全期/两子期）
   ② 落成可复用因子面板（XGBOOST / FNN 的估值错配与直接收益预测）

输出：
  out_v2/dual_filter_report.md        双重过滤结果表
  factors/funda_ml_factor_panel.parquet   可复用因子面板
  factors/README.md                    口径登记（预注册）
"""
import io, sys, os
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

REPRO = r'h:\Claw_backup\财报分析模型\repro'
MM = os.path.join(REPRO, 'out_v2', 'mismatch_panel_v2.parquet')
DIRECT = os.path.join(REPRO, 'out_v2', 'direct_ret_pred.parquet')
PANEL = os.path.join(REPRO, 'data_ext', 'monthly_panel_ext.parquet')
OUT_DIR = os.path.join(REPRO, 'out_v2')
FACTOR_DIR = os.path.join(REPRO, 'factors')
os.makedirs(FACTOR_DIR, exist_ok=True)

FACTORS = {
    'M_XGBOOST': '估值错配-XGBoost',
    'M_FNN': '估值错配-FNN',
    'M_NONL': '估值错配-非线性集成',
    'D_GBDT': '直接收益-GBDT',
    'D_FNN': '直接收益-FNN',
}
Q = 10


# ------------------------------------------------------------------ 工具
def size_neutral(df, col):
    """逐月横截面 M ~ log(V) 取残差。"""
    out = np.full(len(df), np.nan)
    ok = df[col].notna() & df['lV'].notna()
    for mt, g in df.loc[ok].groupby('month'):
        if len(g) < 20:
            continue
        X = np.column_stack([np.ones(len(g)), g['lV'].values])
        y = g[col].values
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        out[g.index.values] = y - X @ beta
    return out


def size_ind_neutral(df, col):
    """逐月横截面 M ~ log(V) + 行业哑变量 取残差。"""
    out = np.full(len(df), np.nan)
    ok = df[col].notna() & df['lV'].notna() & df['sw_industry'].notna()
    for mt, g in df.loc[ok].groupby('month'):
        if len(g) < 50:
            continue
        dums = pd.get_dummies(g['sw_industry'].astype(str), drop_first=True).astype(float)
        if dums.shape[1] > 60:
            dums = dums.iloc[:, :60]
        X = np.column_stack([np.ones(len(g)), g['lV'].values, dums.values])
        y = g[col].values
        try:
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        except Exception:
            continue
        out[g.index.values] = y - X @ beta
    return out


def hl(sub, col, weight='ew'):
    """十分位多空（等权 / 市值加权）。"""
    rows = []
    for mt, g in sub.dropna(subset=[col, 'ret_fwd', 'V']).groupby('month'):
        if len(g) < 50:
            continue
        g = g.copy()
        try:
            g['d'] = pd.qcut(g[col].rank(method='first'), Q, labels=False) + 1
        except Exception:
            continue
        dd = {}
        for k in range(1, Q + 1):
            m = g['d'] == k
            if not m.any():
                continue
            v = g.loc[m, 'ret_fwd'].values
            if weight == 'ew':
                dd[k] = v.mean()
            else:
                w = np.clip(g.loc[m, 'V'].values, 1, None)
                dd[k] = np.average(v, weights=w)
        if Q in dd and 1 in dd:
            rows.append(dd[Q] - dd[1])
    s = pd.Series(rows).dropna()
    if len(s) < 2 or s.std() == 0:
        return np.nan, np.nan, len(s)
    return s.mean() * 100, s.mean() / (s.std() / np.sqrt(len(s))), len(s)


# ------------------------------------------------------------------ 载入
def load():
    df = pd.read_parquet(MM)
    df = df.dropna(subset=['ret_fwd', 'V'])
    df = df[df['V'] > 0].copy()
    df['lV'] = np.log(df['V'])
    # 补行业
    ext = pd.read_parquet(PANEL, columns=['ts_code', 'month_end', 'sw_industry'])
    ext['month'] = pd.to_datetime(ext['month_end']).dt.to_period('M')
    ext = ext.drop_duplicates(['ts_code', 'month'])
    df = df.merge(ext[['ts_code', 'month', 'sw_industry']], on=['ts_code', 'month'], how='left')
    # 合并直接收益预测
    d = pd.read_parquet(DIRECT, columns=['ts_code', 'month', 'D_GBDT', 'D_FNN'])
    d = d.drop_duplicates(['ts_code', 'month'])
    df = df.merge(d, on=['ts_code', 'month'], how='left')
    df['year'] = df['month'].astype(str).str[:4].astype(int)
    return df


def build_neutral(df):
    cols = {}
    for c in FACTORS:
        if c not in df.columns:
            continue
        cols[c + '|size'] = size_neutral(df, c)
        cols[c + '|size_ind'] = size_ind_neutral(df, c)
    return pd.concat([df, pd.DataFrame(cols, index=df.index)], axis=1)


# ------------------------------------------------------------------ 主流程
def main():
    df = load()
    print('样本 %d 行 | %d 个月 | %d 股' % (len(df), df['month'].nunique(), df['ts_code'].nunique()))
    df = build_neutral(df)

    periods = [('全期', df), ('2009-15', df[df.year <= 2015]), ('2016-25', df[df.year >= 2016])]

    lines = ['# 双重过滤稳健性（等权/市值加权 × 规模中性）', '']
    lines.append('样本：%d 行 / %d 个月 / %d 股，2009-02 ~ 2025-05' % (
        len(df), df['month'].nunique(), df['ts_code'].nunique()))
    lines.append('')

    for pname, sub in periods:
        lines.append('## %s（%d 个月）' % (pname, sub['month'].nunique()))
        lines.append('')
        lines.append('| 因子 | 等权 原始 | 等权 规模中性 | 等权 规模+行业 | 市值加权 原始 | 市值加权 规模中性 | 市值加权 规模+行业 |')
        lines.append('|---|---|---|---|---|---|---|')
        for c, label in FACTORS.items():
            if c not in df.columns:
                continue
            vals = {}
            for w in ('ew', 'vw'):
                for tag, col in (('raw', c), ('size', c + '|size'), ('size_ind', c + '|size_ind')):
                    m, t, n = hl(sub, col, w)
                    vals[(w, tag)] = '%+.2f (%.2f)' % (m, t)
            lines.append('| %s | %s | %s | %s | %s | %s | %s |' % (
                label, vals[('ew', 'raw')], vals[('ew', 'size')], vals[('ew', 'size_ind')],
                vals[('vw', 'raw')], vals[('vw', 'size')], vals[('vw', 'size_ind')]))
        lines.append('')

    # 控制台摘要
    print()
    print('=' * 100)
    print('【2016-25 段 · 双重过滤】')
    print('=' * 100)
    sub = df[df.year >= 2016]
    print('%-22s %16s %16s %16s' % ('因子', '等权 原始', '等权 规模中性', '等权 规模+行业'))
    for c, label in FACTORS.items():
        if c not in df.columns:
            continue
        a = hl(sub, c, 'ew'); b = hl(sub, c + '|size', 'ew'); e = hl(sub, c + '|size_ind', 'ew')
        print('%-22s %7.2f(%5.2f) %7.2f(%5.2f) %7.2f(%5.2f)' % (label, a[0], a[1], b[0], b[1], e[0], e[1]))

    print()
    print('=' * 100)
    print('【全期 · 市值加权 × 规模中性】')
    print('=' * 100)
    print('%-22s %16s %16s' % ('因子', '等权 规模中性', '市值加权 规模中性'))
    for c, label in FACTORS.items():
        if c not in df.columns:
            continue
        b = hl(df, c + '|size', 'ew'); f = hl(df, c + '|size', 'vw')
        print('%-22s %7.2f(%5.2f) %7.2f(%5.2f)' % (label, b[0], b[1], f[0], f[1]))

    with open(os.path.join(OUT_DIR, 'dual_filter_report.md'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines))
    print()
    print('已写 out_v2/dual_filter_report.md')

    # ---------------- 因子面板 ----------------
    fp = df[['ts_code', 'month', 'ret_fwd', 'V', 'sw_industry']].copy()
    fp['month_end'] = fp['month'].dt.to_timestamp(how='end').dt.normalize()
    fp['f_mis_xgb'] = df['M_XGBOOST'].values
    fp['f_mis_xgb_sn'] = df['M_XGBOOST|size'].values
    fp['f_mis_fnn'] = df['M_FNN'].values
    fp['f_mis_fnn_sn'] = df['M_FNN|size'].values
    fp['f_dir_gbdt'] = df['D_GBDT'].values
    fp['f_dir_fnn'] = df['D_FNN'].values
    fp = fp[['ts_code', 'month_end', 'f_mis_xgb', 'f_mis_xgb_sn', 'f_mis_fnn', 'f_mis_fnn_sn',
             'f_dir_gbdt', 'f_dir_fnn', 'ret_fwd', 'V', 'sw_industry']]
    out_pq = os.path.join(FACTOR_DIR, 'funda_ml_factor_panel.parquet')
    fp.to_parquet(out_pq, index=False)
    print('已写 %s（%d 行 × %d 列）' % (out_pq, fp.shape[0], fp.shape[1]))

    return df, sub, fp


if __name__ == '__main__':
    main()
