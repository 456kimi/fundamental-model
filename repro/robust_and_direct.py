# -*- coding: utf-8 -*-
"""双重验证 + 直接收益预测基准。

① 稳健性：XGBOOST / M_NONL 在【等权/市值加权】×【原始/规模中性】四种组合下的 H-L。
② 直接基准：同一 37 个基本面变量 + 滚动 5 月窗口，直接预测下月收益（非绕预测估值），
            比较"估值错配路线"与"直接选股路线"。
"""
import io, sys, os, time
import numpy as np
import pandas as pd
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

REPRO = r'h:\Claw_backup\财报分析模型\repro'
PANEL = os.path.join(REPRO, 'data_ext', 'monthly_panel_ext.parquet')
MM = os.path.join(REPRO, 'out_v2', 'mismatch_panel_v2.parquet')
OUT_DIR = os.path.join(REPRO, 'out_v2')

FIN = ['AT', 'SEQ', 'ICAPT', 'PSTK', 'PPENT', 'LT', 'CEQ', 'AO', 'DLTT', 'LO', 'ACO',
       'CHE', 'LCO', 'AP', 'DV', 'SALE', 'XIDO', 'IB', 'NI', 'PI', 'TXT', 'NOPI',
       'DVCASH', 'FINACT', 'PURCHASE', 'RECOVER', 'INVACT', 'EMPLOYEE', 'CASH',
       'GPR', 'ART', 'IT', 'RD', 'CUSTOMER', 'SUPPLIER']
PROXY = ['NEWS', 'MEDIA']
FEATURES = FIN + PROXY
CORE = ['AT', 'LT', 'SEQ']
WIN = 5


# ------------------------------------------------------------ 1) 稳健性四组合
def part1():
    df = pd.read_parquet(MM)
    df = df.dropna(subset=['ret_fwd', 'V'])
    df = df[df['V'] > 0]
    df['lV'] = np.log(df['V'])
    for c in ['M_XGBOOST', 'M_NONL', 'M_FNN']:
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

    def hl(sub, c, weight):
        rows = []
        for mt, g in sub.dropna(subset=[c, 'ret_fwd', 'V']).groupby('month'):
            if len(g) < 50:
                continue
            g = g.copy()
            try:
                g['d'] = pd.qcut(g[c].rank(method='first'), 10, labels=False) + 1
            except Exception:
                continue
            if weight == 'ew':
                dd = g.groupby('d')['ret_fwd'].mean()
            else:
                def vw(x):
                    return np.average(x['ret_fwd'], weights=np.clip(x['V'], 1, None))
                dd = g.groupby('d').apply(vw, include_groups=False)
            if 10 in dd and 1 in dd:
                rows.append(dd[10] - dd[1])
        s = pd.Series(rows).dropna()
        if len(s) < 2 or s.std() == 0:
            return np.nan, np.nan
        return s.mean() * 100, s.mean() / (s.std() / np.sqrt(len(s)))

    print('=' * 74)
    print('[1] 稳健性四组合（H-L 月度 %，t 值）')
    print('=' * 74)
    print('%-14s %22s %22s' % ('模型', '等权 原始→中性', '市值加权 原始→中性'))
    for c in ['M_XGBOOST', 'M_NONL', 'M_FNN']:
        a, ta = hl(df, c, 'ew'); b, tb = hl(df, c + '_res', 'ew')
        e, te = hl(df, c, 'vw'); f, tf = hl(df, c + '_res', 'vw')
        print('%-14s %9.3f(%5.2f)→%6.3f(%5.2f)  %9.3f(%5.2f)→%6.3f(%5.2f)' % (
            c, a, ta, b, tb, e, te, f, tf))
    print()


# ------------------------------------------------------------ 2) 直接收益预测
def part2():
    df = pd.read_parquet(PANEL)
    df['month'] = pd.to_datetime(df['month_end']).dt.to_period('M')
    df = df[df['ts_code'].str.endswith(('.SH', '.SZ'))]
    df = df[~df['ts_code'].str.startswith('688')]
    df = df[df['V'].notna() & (df['V'] > 0) & df['ret_fwd'].notna()]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=CORE)
    df = df[df['SEQ'] > 0]
    for c in FEATURES:
        df[c] = df[c].fillna(df.groupby('month')[c].transform('median'))
        df[c] = df[c].fillna(df[c].median()).fillna(0.0)
    df = df.dropna(subset=FEATURES)
    df['ret_fwd'] = df['ret_fwd'].clip(-0.5, 1.0)
    df['lV'] = np.log(df['V'])
    print('[2] 直接收益预测 | 面板 %d 行 | %s..%s' % (len(df), df['month'].min(), df['month'].max()))

    months = sorted(df['month'].unique())
    recs = []
    t0 = time.time()
    for i in range(WIN, len(months)):
        mt, trm = months[i], months[i - WIN:i]
        tr = df[df['month'].isin(trm)]
        te = df[df['month'] == mt]
        if len(tr) < 200 or len(te) < 30:
            continue
        mu, sd = tr[FEATURES].mean(), tr[FEATURES].std().replace(0, 1.0)
        Xtr = ((tr[FEATURES] - mu) / sd).values.astype(np.float64)
        Xte = ((te[FEATURES] - mu) / sd).values.astype(np.float64)
        ytr = tr['ret_fwd'].values.astype(np.float64)
        out = te[['ts_code', 'month', 'V', 'ret_fwd']].copy()
        out['D_RF'] = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=0).fit(Xtr, ytr).predict(Xte)
        out['D_GBDT'] = HistGradientBoostingRegressor(max_iter=100, random_state=0).fit(Xtr, ytr).predict(Xte)
        out['D_XGB'] = XGBRegressor(n_estimators=100, max_depth=6, n_jobs=-1, verbosity=0, random_state=0).fit(Xtr, ytr).predict(Xte)
        out['D_FNN'] = MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=200, random_state=0).fit(Xtr, ytr).predict(Xte)
        out['D_NONL'] = out[['D_RF', 'D_GBDT', 'D_XGB', 'D_FNN']].mean(axis=1)
        recs.append(out)
        if (i - WIN) % 30 == 0:
            print('   %s (%d/%d) %.0fs' % (mt, i - WIN + 1, len(months) - WIN, time.time() - t0))
    res = pd.concat(recs, ignore_index=True)
    res.to_parquet(os.path.join(OUT_DIR, 'direct_ret_pred.parquet'))

    # 中性化
    res['lV'] = np.log(res['V'])
    for c in ['D_RF', 'D_GBDT', 'D_XGB', 'D_FNN', 'D_NONL']:
        out = np.full(len(res), np.nan)
        ok = res[c].notna() & res['lV'].notna()
        for mt, g in res.loc[ok].groupby('month'):
            if len(g) < 20:
                continue
            X = np.column_stack([np.ones(len(g)), g['lV'].values])
            y = g[c].values
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            out[g.index.values] = y - X @ beta
        res[c + '_res'] = out

    def hl(sub, c, weight):
        rows = []
        for mt, g in sub.dropna(subset=[c, 'ret_fwd', 'V']).groupby('month'):
            if len(g) < 50:
                continue
            g = g.copy()
            try:
                g['d'] = pd.qcut(g[c].rank(method='first'), 10, labels=False) + 1
            except Exception:
                continue
            if weight == 'ew':
                dd = g.groupby('d')['ret_fwd'].mean()
            else:
                def vw(x):
                    return np.average(x['ret_fwd'], weights=np.clip(x['V'], 1, None))
                dd = g.groupby('d').apply(vw, include_groups=False)
            if 10 in dd and 1 in dd:
                rows.append(dd[10] - dd[1])
        s = pd.Series(rows).dropna()
        if len(s) < 2 or s.std() == 0:
            return np.nan, np.nan
        return s.mean() * 100, s.mean() / (s.std() / np.sqrt(len(s)))

    print()
    print('===== [2] 直接预测下月收益（等权 / 市值加权，原始 / 中性化）=====')
    print('%-10s %20s %20s' % ('模型', '等权 原始→中性', '市值加权 原始→中性'))
    for c in ['D_RF', 'D_GBDT', 'D_XGB', 'D_FNN', 'D_NONL']:
        a, ta = hl(res, c, 'ew'); b, tb = hl(res, c + '_res', 'ew')
        e, te = hl(res, c, 'vw'); f, tf = hl(res, c + '_res', 'vw')
        print('%-10s %9.3f(%5.2f)→%6.3f(%5.2f)  %9.3f(%5.2f)→%6.3f(%5.2f)' % (
            c, a, ta, b, tb, e, te, f, tf))

    # 对比: 估值错配路线 M_XGBOOST vs 直接路线 D_XGB（同集合、同窗口）
    print()
    print('===== 路线对比（等权 / 原始）=====')
    print('   估值错配 XGBOOST: 见 [1]')
    print('   直接收益 D_XGB  : 见上表')


if __name__ == '__main__':
    part1()
    part2()
    print('done')
