# -*- coding: utf-8 -*-
import os, sys, io, time, json
import numpy as np
import pandas as pd
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

BASE = r'h:\Claw_backup'
PANEL = os.path.join(BASE, 'fin_quant', 'data', 'fin_factor_panel_v1.parquet')
DAILY = os.path.join(BASE, 'vp_probe_data', 'stock_pv.parquet')
OUT = os.path.join(BASE, '财报分析模型', 'repro', 'out')
os.makedirs(OUT, exist_ok=True)

FEATURES = [
    'total_assets', 'total_liab', 'total_hldr_eqy_exc_min_int',
    'total_cur_assets', 'total_cur_liab', 'total_share',
    'ni_ttm', 'rev_ttm', 'cost_ttm', 'trev_ttm', 'cfo_ttm',
    'roe', 'roe_dt', 'grossprofit_margin', 'debt_to_assets',
    'current_ratio', 'or_yoy', 'roe_yoy', 'netprofit_yoy',
]

WIN = 5
MODELS = ['OLS', 'RIDGE', 'LASSO', 'ELASTICNET', 'PCR',
          'DT', 'RF', 'GBDT', 'XGBOOST', 'FNN']
LIN = ['OLS', 'RIDGE', 'LASSO', 'ELASTICNET', 'PCR']
NONL = ['DT', 'RF', 'GBDT', 'XGBOOST', 'FNN']


def log(m):
    print('[%s] %s' % (time.strftime('%H:%M:%S'), m), flush=True)


def load_panel():
    log('load panel')
    df = pd.read_parquet(PANEL)
    df['month'] = pd.to_datetime(df['month_end']).dt.to_period('M')
    df['V'] = df['me_close'] * df['total_share']
    keep = ['ts_code', 'month', 'V', 'is_one_word_me', 'age_ok', 'is_bank'] + FEATURES
    df = df[[c for c in keep if c in df.columns]].copy()
    return df


def load_monthly_returns():
    log('load daily pv -> monthly returns')
    pv = pd.read_parquet(DAILY, columns=['ts_code', 'trade_date', 'pct_chg'])
    pv['month'] = pd.to_datetime(pv['trade_date']).dt.to_period('M')
    pv['g'] = 1.0 + pv['pct_chg'].fillna(0) / 100.0
    mret = pv.groupby(['ts_code', 'month'], observed=True)['g'].prod().reset_index()
    mret['mret'] = mret['g'] - 1.0
    mret = mret[['ts_code', 'month', 'mret']].sort_values(['ts_code', 'month'])
    mret['ret_fwd'] = mret.groupby('ts_code')['mret'].shift(-1)
    return mret


def build_panel():
    df = load_panel()
    mret = load_monthly_returns()
    df = df.merge(mret[['ts_code', 'month', 'ret_fwd']], on=['ts_code', 'month'], how='left')
    df = df[df['V'].notna() & (df['V'] > 0)]
    df = df[df['total_hldr_eqy_exc_min_int'].fillna(-1) > 0]
    if 'age_ok' in df.columns:
        df = df[df['age_ok'].fillna(True)]
    df = df[~df['is_one_word_me'].fillna(False).astype(bool)]
    df = df.dropna(subset=FEATURES + ['ret_fwd'])
    for f in FEATURES:
        q1, q99 = df.groupby('month')[f].transform(lambda s: s.quantile(0.01)), \
                  df.groupby('month')[f].transform(lambda s: s.quantile(0.99))
        df[f] = df[f].clip(q1, q99)
    df['ret_fwd'] = df['ret_fwd'].clip(-0.5, 1.0)
    df['y'] = np.log(df['V'])
    log('panel rows %d, months %d, stocks %d' % (len(df), df['month'].nunique(), df['ts_code'].nunique()))
    return df


def fit_predict(name, Xtr, ytr, Xte):
    rng = 0
    if name == 'OLS':
        m = LinearRegression().fit(Xtr, ytr)
    elif name == 'RIDGE':
        m = Ridge(alpha=1.0).fit(Xtr, ytr)
    elif name == 'LASSO':
        m = Lasso(alpha=1e-3, max_iter=5000).fit(Xtr, ytr)
    elif name == 'ELASTICNET':
        m = ElasticNet(alpha=1e-3, l1_ratio=0.5, max_iter=5000).fit(Xtr, ytr)
    elif name == 'PCR':
        k = min(8, Xtr.shape[1])
        p = PCA(n_components=k, random_state=rng).fit(Xtr)
        m = LinearRegression().fit(p.transform(Xtr), ytr)
        return m.predict(p.transform(Xte))
    elif name == 'DT':
        m = DecisionTreeRegressor(random_state=rng).fit(Xtr, ytr)
    elif name == 'RF':
        m = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=rng).fit(Xtr, ytr)
    elif name == 'GBDT':
        m = HistGradientBoostingRegressor(max_iter=100, random_state=rng).fit(Xtr, ytr)
    elif name == 'XGBOOST':
        m = XGBRegressor(n_estimators=100, max_depth=6, n_jobs=-1,
                         verbosity=0, random_state=rng).fit(Xtr, ytr)
    elif name == 'FNN':
        m = MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=200,
                         random_state=rng).fit(Xtr, ytr)
    else:
        raise ValueError(name)
    return m.predict(Xte)


def run_rolling(df):
    months = sorted(df['month'].unique())
    recs = []
    t0 = time.time()
    for i in range(WIN, len(months)):
        mt = months[i]
        tr_months = months[i - WIN:i]
        tr = df[df['month'].isin(tr_months)]
        te = df[df['month'] == mt]
        if len(tr) < 200 or len(te) < 30:
            continue
        mu = tr[FEATURES].mean()
        sd = tr[FEATURES].std().replace(0, 1.0)
        Xtr = ((tr[FEATURES] - mu) / sd).values.astype(np.float64)
        Xte = ((te[FEATURES] - mu) / sd).values.astype(np.float64)
        ytr = tr['y'].values.astype(np.float64)
        out = te[['ts_code', 'month', 'V', 'ret_fwd']].copy()
        for nm in MODELS:
            try:
                pred_logv = fit_predict(nm, Xtr, ytr, Xte)
                pv = np.exp(pred_logv)
                out['M_' + nm] = (pv - out['V'].values) / out['V'].values
            except Exception as e:
                log('  %s %s FAIL %s' % (mt, nm, str(e)[:80]))
                out['M_' + nm] = np.nan
        out['M_L'] = out[['M_' + n for n in LIN]].mean(axis=1)
        out['M_NONL'] = out[['M_' + n for n in NONL]].mean(axis=1)
        out['M_COM'] = out[['M_' + n for n in MODELS]].mean(axis=1)
        recs.append(out)
        if (i - WIN) % 10 == 0:
            log('  month %s done (%d/%d) elapsed %.0fs' % (mt, i - WIN + 1, len(months) - WIN, time.time() - t0))
    res = pd.concat(recs, ignore_index=True)
    return res


def portfolio(res, cols):
    rows = []
    for c in cols:
        sub = res.dropna(subset=[c, 'ret_fwd']).copy()
        hl = []
        deciles = {}
        for mt, g in sub.groupby('month'):
            if len(g) < 50:
                continue
            try:
                g = g.copy()
                g['dec'] = pd.qcut(g[c].rank(method='first'), 10, labels=False) + 1
            except Exception:
                continue
            d = g.groupby('dec')['ret_fwd'].mean()
            for k, v in d.items():
                deciles.setdefault(k, []).append(v)
            hl.append(d.get(10, np.nan) - d.get(1, np.nan))
        hl = pd.Series(hl).dropna()
        mean = hl.mean()
        tstat = mean / (hl.std() / np.sqrt(len(hl))) if len(hl) > 1 and hl.std() > 0 else np.nan
        ann = (1 + mean) ** 12 - 1
        drow = {'model': c, 'H-L monthly %': round(mean * 100, 4),
                't': round(tstat, 4), 'ann %': round(ann * 100, 2), 'N_months': len(hl)}
        for k in range(1, 11):
            if k in deciles:
                drow['D%d' % k] = round(np.mean(deciles[k]) * 100, 4)
        rows.append(drow)
    return pd.DataFrame(rows).set_index('model')


def main():
    df = build_panel()
    res = run_rolling(df)
    res.to_parquet(os.path.join(OUT, 'mismatch_panel.parquet'))
    allcols = ['M_' + n for n in MODELS] + ['M_L', 'M_NONL', 'M_COM']
    desc = res[allcols].describe().T[['count', 'mean', 'std', 'min', '50%', 'max']]
    desc.columns = ['N', 'Mean', 'SD', 'Min', 'Median', 'Max']
    desc.to_csv(os.path.join(OUT, 'desc_stats.csv'), encoding='utf-8-sig')
    log('=== descriptive stats ===')
    print(desc.round(4).to_string(), flush=True)
    pf = portfolio(res, allcols)
    pf.to_csv(os.path.join(OUT, 'portfolio_hl.csv'), encoding='utf-8-sig')
    log('=== H-L portfolio ===')
    print(pf[['H-L monthly %', 't', 'ann %', 'N_months']].to_string(), flush=True)
    log('done')


if __name__ == '__main__':
    main()
