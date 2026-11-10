# -*- coding: utf-8 -*-
"""v2 复现：扩展变量集（35 财务 + 2 代理）滚动 10 模型 + 十分位多空。"""
import os, sys, io, time
import numpy as np
import pandas as pd
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

REPRO = r'h:\Claw_backup\财报分析模型\repro'
PANEL = os.path.join(REPRO, 'data_ext', 'monthly_panel_ext.parquet')
OUT = os.path.join(REPRO, 'out_v2')
os.makedirs(OUT, exist_ok=True)

FIN = ['AT', 'SEQ', 'ICAPT', 'PSTK', 'PPENT', 'LT', 'CEQ', 'AO', 'DLTT', 'LO', 'ACO',
       'CHE', 'LCO', 'AP', 'DV', 'SALE', 'XIDO', 'IB', 'NI', 'PI', 'TXT', 'NOPI',
       'DVCASH', 'FINACT', 'PURCHASE', 'RECOVER', 'INVACT', 'EMPLOYEE', 'CASH',
       'GPR', 'ART', 'IT', 'RD', 'CUSTOMER', 'SUPPLIER']
PROXY = ['NEWS', 'MEDIA']
FEATURES = FIN + PROXY

WIN = 5
MODELS = ['OLS', 'RIDGE', 'LASSO', 'ELASTICNET', 'PCR', 'DT', 'RF', 'GBDT', 'XGBOOST', 'FNN']
LIN = ['OLS', 'RIDGE', 'LASSO', 'ELASTICNET', 'PCR']
NONL = ['DT', 'RF', 'GBDT', 'XGBOOST', 'FNN']


def log(m):
    print('[%s] %s' % (time.strftime('%H:%M:%S'), m), flush=True)


def fit_predict(name, Xtr, ytr, Xte):
    r = 0
    if name == 'OLS':
        m = LinearRegression().fit(Xtr, ytr)
    elif name == 'RIDGE':
        m = Ridge(alpha=1.0).fit(Xtr, ytr)
    elif name == 'LASSO':
        m = Lasso(alpha=1e-2, max_iter=500, tol=1e-2).fit(Xtr, ytr)
    elif name == 'ELASTICNET':
        m = ElasticNet(alpha=1e-2, l1_ratio=0.5, max_iter=500, tol=1e-2).fit(Xtr, ytr)
    elif name == 'PCR':
        p = PCA(n_components=min(10, Xtr.shape[1]), random_state=r).fit(Xtr)
        m = LinearRegression().fit(p.transform(Xtr), ytr)
        return m.predict(p.transform(Xte))
    elif name == 'DT':
        m = DecisionTreeRegressor(random_state=r).fit(Xtr, ytr)
    elif name == 'RF':
        m = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=r).fit(Xtr, ytr)
    elif name == 'GBDT':
        m = HistGradientBoostingRegressor(max_iter=100, random_state=r).fit(Xtr, ytr)
    elif name == 'XGBOOST':
        m = XGBRegressor(n_estimators=100, max_depth=6, n_jobs=-1, verbosity=0, random_state=r).fit(Xtr, ytr)
    elif name == 'FNN':
        m = MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=200, random_state=r).fit(Xtr, ytr)
    return m.predict(Xte)


CORE = ['AT', 'LT', 'SEQ']


def load():
    df = pd.read_parquet(PANEL)
    df['month'] = pd.to_datetime(df['month_end']).dt.to_period('M')
    # 宇宙: 沪深两市 + 剔除科创板(688) —— 与论文一致; 排除北交所/新三板(无涨跌幅限制)
    df = df[df['ts_code'].str.endswith(('.SH', '.SZ'))]
    df = df[~df['ts_code'].str.startswith('688')]
    df = df[df['V'].notna() & (df['V'] > 0) & df['ret_fwd'].notna()]
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=CORE)
    # 剔除净资产为负 (论文黑名单)
    df = df[df['SEQ'] > 0]
    n0 = len(df)
    # 缺失填充: 按月横截面中位数 -> 全样本中位数 -> 0
    # (PSTK 优先股 2009-01~2014-03 未披露、RD 研发 2018 前未单列, 早期整列为空 -> 填 0)
    for c in FEATURES:
        df[c] = df[c].fillna(df.groupby('month')[c].transform('median'))
        df[c] = df[c].fillna(df[c].median())
        df[c] = df[c].fillna(0.0)
    df = df.dropna(subset=FEATURES)
    log('面板 %d 行 (核心筛选 %d -> 填充后 %d) | 股票 %d | %s..%s' % (
        len(df), n0, len(df), df['ts_code'].nunique(), df['month'].min(), df['month'].max()))
    df['ret_fwd'] = df['ret_fwd'].clip(-0.5, 1.0)
    df['y'] = np.log(df['V'])
    return df


def run(df):
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
        ytr = tr['y'].values.astype(np.float64)
        out = te[['ts_code', 'month', 'V', 'ret_fwd']].copy()
        for nm in MODELS:
            try:
                pv = np.exp(np.clip(fit_predict(nm, Xtr, ytr, Xte), -20, 30))
                pv = np.clip(pv, 0.01 * out['V'].values, 20 * out['V'].values)   # 防外推爆炸
                out['M_' + nm] = (pv - out['V'].values) / out['V'].values
            except Exception as e:
                log('  %s %s FAIL %s' % (mt, nm, str(e)[:60]))
                out['M_' + nm] = np.nan
        out['M_L'] = out[['M_' + n for n in LIN]].mean(axis=1)
        out['M_NONL'] = out[['M_' + n for n in NONL]].mean(axis=1)
        out['M_COM'] = out[['M_' + n for n in MODELS]].mean(axis=1)
        recs.append(out)
        if (i - WIN) % 10 == 0:
            log('  %s (%d/%d) %.0fs' % (mt, i - WIN + 1, len(months) - WIN, time.time() - t0))
    return pd.concat(recs, ignore_index=True)


def portfolio(res, cols):
    rows = []
    for c in cols:
        sub = res.dropna(subset=[c, 'ret_fwd'])
        hl, dec = [], {}
        for mt, g in sub.groupby('month'):
            if len(g) < 50:
                continue
            g = g.copy()
            g['d'] = pd.qcut(g[c].rank(method='first'), 10, labels=False) + 1
            d = g.groupby('d')['ret_fwd'].mean()
            for k, v in d.items():
                dec.setdefault(k, []).append(v)
            hl.append(d.get(10, np.nan) - d.get(1, np.nan))
        hl = pd.Series(hl).dropna()
        mean = hl.mean()
        t = mean / (hl.std() / np.sqrt(len(hl))) if len(hl) > 1 and hl.std() > 0 else np.nan
        row = {'model': c, 'HL%': round(mean * 100, 4), 't': round(t, 4),
               'ann%': round(((1 + mean) ** 12 - 1) * 100, 2), 'N': len(hl)}
        for k in range(1, 11):
            if k in dec:
                row['D%d' % k] = round(np.mean(dec[k]) * 100, 4)
        rows.append(row)
    return pd.DataFrame(rows).set_index('model')


def main():
    df = load()
    res = run(df)
    res.to_parquet(os.path.join(OUT, 'mismatch_panel_v2.parquet'))
    cols = ['M_' + n for n in MODELS] + ['M_L', 'M_NONL', 'M_COM']
    pf = portfolio(res, cols)
    pf.to_csv(os.path.join(OUT, 'portfolio_v2.csv'), encoding='utf-8-sig')
    log('=== v2 多空组合（37 变量）===')
    print(pf[['HL%', 't', 'ann%', 'N']].to_string(), flush=True)
    log('done')


if __name__ == '__main__':
    main()
