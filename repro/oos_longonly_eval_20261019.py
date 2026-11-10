# -*- coding: utf-8 -*-
"""财报分析模型：① 多头超额口径（相对全市场等权 / 中证1000）
                 ② 2025-06~2026-09 样本外（OOS）检验

OOS 因子生成严格复刻 repro_funda_v2.py / robust_and_direct.py：
  估值错配  y=log(V)，5 月滚动重训，M=(pred_V - V)/V，规模中性化取残差
  直接收益  y=ret_fwd，5 月滚动重训
  模型      XGB(n=100,d=6) / FNN((32,16),200) / HistGBDT(100)
与样本内唯一的差别（防前视）：缺失值填充的"全样本中位数"只用 2025-05 及以前估计。

输出：
  out_v2/oos_factor_panel.parquet
  out_v2/longonly_oos_report_20261019.md
  factors/longonly_excess.csv / factors/oos_metrics.csv
"""
import io
import os
import sys
import time
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

REPRO = r'h:\Claw_backup\财报分析模型\repro'
OOS_DIR = os.path.join(REPRO, 'data_oos')
OUT_DIR = os.path.join(REPRO, 'out_v2')
FACTOR_DIR = os.path.join(REPRO, 'factors')
OLD_PANEL = os.path.join(REPRO, 'data_ext', 'monthly_panel_ext.parquet')
OOS_PANEL = os.path.join(OOS_DIR, 'monthly_panel_oos.parquet')
IS_FACTOR = os.path.join(FACTOR_DIR, 'funda_ml_factor_panel.parquet')
IDX_P620 = r'h:\Claw_backup\panel620\benchmarks\index_daily.parquet'
IDX_OOS = os.path.join(OOS_DIR, 'index_000852.parquet')

FIN = ['AT', 'SEQ', 'ICAPT', 'PSTK', 'PPENT', 'LT', 'CEQ', 'AO', 'DLTT', 'LO', 'ACO',
       'CHE', 'LCO', 'AP', 'DV', 'SALE', 'XIDO', 'IB', 'NI', 'PI', 'TXT', 'NOPI',
       'DVCASH', 'FINACT', 'PURCHASE', 'RECOVER', 'INVACT', 'EMPLOYEE', 'CASH',
       'GPR', 'ART', 'IT', 'RD', 'CUSTOMER', 'SUPPLIER']
PROXY = ['NEWS', 'MEDIA']
FEATURES = FIN + PROXY
CORE = ['AT', 'LT', 'SEQ']
WIN = 5
COST = 0.0020
ENTRY_PCT, HOLD_PCT = 0.10, 0.25
IS_CUTOFF = pd.Period('2025-05', 'M')
OOS_START = pd.Period('2025-06', 'M')
OOS_EVAL_END = pd.Period('2026-08', 'M')     # 2026-09 的 ret_fwd 需 10 月收益，尚未完成
FACTORS = [('f_dir_fnn', '直接收益-FNN'), ('f_dir_gbdt', '直接收益-GBDT'),
           ('f_mis_xgb_sn', '错配XGB(规模中性)'), ('f_mis_fnn_sn', '错配FNN(规模中性)')]


def log(m=''):
    print(m, flush=True)


def sec(t):
    log('\n' + '=' * 78)
    log('### ' + t)
    log('=' * 78)


# ------------------------------------------------------------------ 数据
def load_combined():
    """合并样本内 + OOS 原始面板，复刻 load() 的宇宙与填充。"""
    old = pd.read_parquet(OLD_PANEL)
    new = pd.read_parquet(OOS_PANEL)
    old['month'] = pd.to_datetime(old['month_end']).dt.to_period('M')
    new['month'] = pd.to_datetime(new['month_end']).dt.to_period('M')
    # OOS 面板从 2025-06 起；旧面板 2025-06 行的 ret_fwd 为空，用新的替换
    old = old[old['month'] < OOS_START]
    df = pd.concat([old, new], ignore_index=True)
    df['month_end'] = pd.to_datetime(df['month_end'])

    df = df[df['ts_code'].str.endswith(('.SH', '.SZ'))]
    df = df[~df['ts_code'].str.startswith('688')]
    df = df[df['V'].notna() & (df['V'] > 0)]
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=CORE)
    df = df[df['SEQ'] > 0]

    # 填充：当月横截面中位数 -> 样本内(<=2025-05)中位数 -> 0   （防前视）
    is_mask = df['month'] <= IS_CUTOFF
    for c in FEATURES:
        df[c] = df[c].fillna(df.groupby('month')[c].transform('median'))
        m_is = df.loc[is_mask, c].median()
        df[c] = df[c].fillna(m_is).fillna(0.0)
    df = df.dropna(subset=FEATURES)
    df['ret_fwd'] = df['ret_fwd'].clip(-0.5, 1.0)
    df['y'] = np.log(df['V'])
    df['lV'] = np.log(df['V'])
    return df


# ------------------------------------------------------------------ OOS 建模
def _fit(name, Xtr, ytr, Xte):
    if name == 'XGB':
        m = XGBRegressor(n_estimators=100, max_depth=6, n_jobs=-1, verbosity=0, random_state=0)
    elif name == 'GBDT':
        m = HistGradientBoostingRegressor(max_iter=100, random_state=0)
    elif name == 'FNN':
        m = MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=200, random_state=0)
    m.fit(Xtr, ytr)
    return m.predict(Xte)


def size_neutral_by_month(df, col):
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


def gen_oos(df, months_target):
    """对 OOS 每个月用前 5 个月重训，产出 4 个因子。"""
    all_months = sorted(df['month'].unique())
    recs = []
    t0 = time.time()
    for mt in months_target:
        idx = all_months.index(mt)
        trm = all_months[max(0, idx - WIN):idx]
        tr = df[df['month'].isin(trm) & df['ret_fwd'].notna()]
        te = df[df['month'] == mt]
        if len(tr) < 200 or len(te) < 30:
            log('  %s 样本不足 (train=%d test=%d)，跳过' % (mt, len(tr), len(te)))
            continue
        mu, sd = tr[FEATURES].mean(), tr[FEATURES].std().replace(0, 1.0)
        Xtr = ((tr[FEATURES] - mu) / sd).values.astype(np.float64)
        Xte = ((te[FEATURES] - mu) / sd).values.astype(np.float64)

        out = te[['ts_code', 'month', 'V', 'ret_fwd']].copy()
        # 估值错配路线
        ytr = tr['y'].values.astype(np.float64)
        for nm, key in [('XGB', 'M_XGBOOST'), ('FNN', 'M_FNN')]:
            pv = np.exp(np.clip(_fit(nm, Xtr, ytr, Xte), -20, 30))
            pv = np.clip(pv, 0.01 * out['V'].values, 20 * out['V'].values)
            out[key] = (pv - out['V'].values) / out['V'].values
        # 直接收益路线
        ytr2 = tr['ret_fwd'].values.astype(np.float64)
        out['D_GBDT'] = _fit('GBDT', Xtr, ytr2, Xte)
        out['D_FNN'] = _fit('FNN', Xtr, ytr2, Xte)
        recs.append(out)
        log('  %s train=%d(%s..%s) test=%d  %.0fs'
            % (mt, len(tr), trm[0], trm[-1], len(te), time.time() - t0))
    res = pd.concat(recs, ignore_index=True)
    res['lV'] = np.log(res['V'])
    res['f_mis_xgb'] = res['M_XGBOOST']
    res['f_mis_fnn'] = res['M_FNN']
    res['f_dir_gbdt'] = res['D_GBDT']
    res['f_dir_fnn'] = res['D_FNN']
    res['f_mis_xgb_sn'] = size_neutral_by_month(res, 'M_XGBOOST')
    res['f_mis_fnn_sn'] = size_neutral_by_month(res, 'M_FNN')
    res['year'] = res['month'].astype(str).str[:4].astype(int)
    return res


# ------------------------------------------------------------------ 基准
def build_index_bench():
    parts = []
    for p in [IDX_P620, IDX_OOS]:
        if not os.path.exists(p):
            continue
        d = pd.read_parquet(p)
        d['trade_date'] = pd.to_datetime(d['trade_date'])
        d = d[d['ts_code'] == '000852.SH']
        if len(d):
            parts.append(d[['trade_date', 'close']])
    idx = pd.concat(parts, ignore_index=True).drop_duplicates('trade_date').sort_values('trade_date')
    idx['month'] = idx['trade_date'].dt.to_period('M')
    me = idx.groupby('month').tail(1)[['month', 'trade_date', 'close']]
    me['ret_idx'] = me['close'] / me['close'].shift(1) - 1
    log('  中证1000: %d 个月 | %s..%s' % (
        len(me), me['month'].min(), me['month'].max()))
    return me[['month', 'ret_idx']]


# ------------------------------------------------------------------ 评估
def smooth(df, col):
    d = df.sort_values(['ts_code', 'month']).copy()
    d[col + '_s3'] = d.groupby('ts_code')[col].transform(
        lambda s: s.rolling(3, min_periods=1).mean())
    return d


def longer_metrics(fac, col, idx_ret):
    """多头组合（建仓10%/持有25%）相对 全市场等权 与 中证1000 的超额。"""
    months = sorted(fac['month'].unique())
    prev = None
    rows = []
    for mt in months:
        g = fac[fac['month'] == mt].dropna(subset=[col, 'ret_fwd'])
        if len(g) < 50:
            continue
        g = g.copy()
        g['rk'] = g[col].rank(method='first')
        n_long = max(int(round(len(g) * ENTRY_PCT)), 5)
        n_hold = max(int(round(len(g) * HOLD_PCT)), n_long)
        top = list(g.sort_values('rk', ascending=False)['ts_code'])
        pool = set(top[:n_hold])
        hold = set(top[:n_long])
        to = 0.0
        if prev is not None:
            keep = prev & pool
            add = [c for c in top if c not in keep][:max(n_long - len(keep), 0)]
            hold = set(list(keep) + add)
            to = 1 - len(prev & hold) / max(len(prev), 1)
        r_long = g.loc[g['ts_code'].isin(hold), 'ret_fwd'].mean()
        r_ew = g['ret_fwd'].mean()                    # 全市场等权（同一候选池）
        r_idx = idx_ret.get(mt, np.nan)
        rows.append({'month': mt, 'long': r_long, 'ew': r_ew, 'idx': r_idx, 'turnover': to})
        prev = hold
    r = pd.DataFrame(rows)
    if r.empty:
        return None
    r['ex_ew'] = r['long'] - r['ew']
    r['ex_idx'] = r['long'] - r['idx']
    r['net_ex_ew'] = r['ex_ew'] - r['turnover'] * 2 * COST
    r['net_ex_idx'] = r['ex_idx'] - r['turnover'] * 2 * COST

    def st(s):
        s = s.dropna()
        if len(s) < 3 or s.std() == 0:
            return np.nan, np.nan
        return s.mean(), s.mean() / (s.std() / np.sqrt(len(s)))
    out = {'n': len(r.dropna(subset=['ex_ew']))}
    for k in ['long', 'ew', 'idx', 'ex_ew', 'ex_idx', 'net_ex_ew', 'net_ex_idx']:
        m, t = st(r[k])
        out[k + '_m'] = m
        out[k + '_t'] = t
    out['turnover'] = r['turnover'].mean()
    return out


def hl_metrics(fac, col):
    rows = []
    for mt, g in fac.dropna(subset=[col, 'ret_fwd']).groupby('month'):
        if len(g) < 50:
            continue
        g = g.copy()
        try:
            g['d'] = pd.qcut(g[col].rank(method='first'), 10, labels=False) + 1
        except Exception:
            continue
        dd = g.groupby('d')['ret_fwd'].mean()
        if 10 in dd.index and 1 in dd.index:
            rows.append(dd[10] - dd[1])
    s = pd.Series(rows).dropna()
    if len(s) < 3 or s.std() == 0:
        return np.nan, np.nan
    return s.mean(), s.mean() / (s.std() / np.sqrt(len(s)))


def rank_ic(fac, col):
    rows = []
    for mt, g in fac.dropna(subset=[col, 'ret_fwd']).groupby('month'):
        if len(g) < 30:
            continue
        rows.append(g[col].corr(g['ret_fwd'], method='spearman'))
    s = pd.Series(rows).dropna()
    if len(s) < 3 or s.std() == 0:
        return np.nan, np.nan, len(s)
    return s.mean(), s.mean() / (s.std() / np.sqrt(len(s))), len(s)


# ------------------------------------------------------------------ 主流程
def main():
    sec('0. 载入数据')
    df = load_combined()
    log('  合并面板: %d 行 | 股票 %d | %s..%s' % (
        len(df), df['ts_code'].nunique(), df['month'].min(), df['month'].max()))
    is_factor = pd.read_parquet(IS_FACTOR)
    is_factor['month'] = pd.to_datetime(is_factor['month_end']).dt.to_period('M')
    log('  样本内因子面板: %d 行 | %s..%s' % (
        len(is_factor), is_factor['month'].min(), is_factor['month'].max()))

    sec('1. 生成 OOS 因子 (2025-06 .. 2026-09)')
    targets = [m for m in sorted(df['month'].unique()) if m >= OOS_START]
    log('  目标月份: %s .. %s (%d 个月)' % (targets[0], targets[-1], len(targets)))
    oos = gen_oos(df, targets)
    oos.to_parquet(os.path.join(OUT_DIR, 'oos_factor_panel.parquet'), index=False)
    log('\n  OOS 因子面板: %d 行 | %s..%s' % (len(oos), oos['month'].min(), oos['month'].max()))

    sec('2. 基准')
    idx = build_index_bench().set_index('month')['ret_idx'].to_dict()

    # 合并 IS + OOS 因子
    isf = is_factor[['ts_code', 'month', 'f_mis_xgb', 'f_mis_xgb_sn', 'f_mis_fnn',
                     'f_mis_fnn_sn', 'f_dir_gbdt', 'f_dir_fnn', 'ret_fwd', 'V']].copy()
    isf['src'] = 'IS'
    o = oos[['ts_code', 'month', 'f_mis_xgb', 'f_mis_xgb_sn', 'f_mis_fnn',
             'f_mis_fnn_sn', 'f_dir_gbdt', 'f_dir_fnn', 'ret_fwd', 'V']].copy()
    o['src'] = 'OOS'
    fac = pd.concat([isf, o], ignore_index=True)

    # 平滑必须在整条序列上算（OOS 首月要用 2025-04/05 的历史），再切区间
    fac = fac.sort_values(['ts_code', 'month'])
    for col, _ in FACTORS:
        fac[col + '_s3'] = fac.groupby('ts_code')[col].transform(
            lambda s: s.rolling(3, min_periods=1).mean())
    fac = fac.sort_index()

    periods = [
        ('样本内 2016-25', fac[(fac['src'] == 'IS') & (fac['month'].astype(str) >= '2016-01')]),
        ('样本内 全期 2009-25', fac[fac['src'] == 'IS']),
        ('样本外 2025-06~2026-08', fac[(fac['src'] == 'OOS') &
                                        (fac['month'] >= OOS_START) & (fac['month'] <= OOS_EVAL_END)]),
    ]

    sec('3. 多头超额（相对 全市场等权 / 中证1000），单位 %/月')
    recs = []
    for pname, sub in periods:
        log('\n--- %s (%d 个月) ---' % (pname, sub['month'].nunique()))
        log('%-20s %8s %8s %8s %7s %8s %7s %7s' % (
            '因子', '多头', '全市场等权', '超额(EW)', 't', '超额(1000)', 't', '换手%/月'))
        for col, label in FACTORS:
            m = longer_metrics(sub, col + '_s3', idx)
            if m is None:
                log('%-20s 样本不足' % label)
                continue
            log('%-20s %8.3f %8.3f %8.3f %7.2f %8.3f %7.2f %7.1f' % (
                label, m['long_m'] * 100, m['ew_m'] * 100, m['ex_ew_m'] * 100,
                m['ex_ew_t'], m['ex_idx_m'] * 100, m['ex_idx_t'], m['turnover'] * 100))
            recs.append({'区间': pname, '因子': label, **{k: v for k, v in m.items()}})
    lo = pd.DataFrame(recs)
    lo.to_csv(os.path.join(FACTOR_DIR, 'longonly_excess.csv'), index=False, encoding='utf-8-sig')

    sec('4. 多空价差 与 RankIC（对照，确认 OOS 口径与样本内同源）')
    mrecs = []
    for pname, sub in periods:
        log('\n--- %s ---' % pname)
        log('%-20s %10s %8s | %10s %8s %6s' % ('因子', '多空%/月', 't', 'RankIC', 'IC_t', '月数'))
        for col, label in FACTORS:
            hm, ht = hl_metrics(sub, col + '_s3')
            ic, ict, n = rank_ic(sub, col + '_s3')
            log('%-20s %10.3f %8.2f | %10.4f %8.2f %6d' % (label, hm * 100, ht, ic, ict, n))
            mrecs.append({'区间': pname, '因子': label, '多空': hm, '多空t': ht,
                          'RankIC': ic, 'IC_t': ict, '月数': n})
    mm = pd.DataFrame(mrecs)
    mm.to_csv(os.path.join(FACTOR_DIR, 'oos_metrics.csv'), index=False, encoding='utf-8-sig')

    # ---------------- 报告 ----------------
    write_report(lo, mm, oos, df)
    log('\n完成。')


def write_report(lo, mm, oos, df):
    L = []
    L.append('# 多头超额口径 + 样本外检验（2025-06 ~ 2026-09）')
    L.append('')
    L.append('> 生成：2026-10-19　脚本：`tmp_run/funda_longonly_oos_20261019.py`')
    L.append('> 口径：信号平滑 3M + 缓冲带（建仓 10% / 持有 25%），单边成本 20bp。')
    L.append('> 多头 = 因子值最高 10% 等权；基准 = ①同池全市场等权 ②中证1000(000852.SH)。')
    L.append('')
    L.append('## 一、多头超额（%/月）')
    L.append('')
    for pname in lo['区间'].unique():
        s = lo[lo['区间'] == pname]
        L.append('### %s' % pname)
        L.append('')
        L.append('| 因子 | 多头 | 全市场等权 | 超额(EW) | t | 超额(中证1000) | t | 月换手% |')
        L.append('|---|---|---|---|---|---|---|---|')
        for _, r in s.iterrows():
            L.append('| %s | %+.3f | %+.3f | %+.3f | %+.2f | %+.3f | %+.2f | %.1f |' % (
                r['因子'], r['long_m'] * 100, r['ew_m'] * 100, r['ex_ew_m'] * 100,
                r['ex_ew_t'], r['ex_idx_m'] * 100, r['ex_idx_t'], r['turnover'] * 100))
        L.append('')
    L.append('## 二、多空价差与 RankIC（对照）')
    L.append('')
    for pname in mm['区间'].unique():
        s = mm[mm['区间'] == pname]
        L.append('### %s' % pname)
        L.append('')
        L.append('| 因子 | 多空%/月 | t | RankIC | IC_t | 月数 |')
        L.append('|---|---|---|---|---|---|')
        for _, r in s.iterrows():
            L.append('| %s | %+.3f | %+.2f | %+.4f | %+.2f | %d |' % (
                r['因子'], r['多空'] * 100, r['多空t'], r['RankIC'], r['IC_t'], r['月数']))
        L.append('')
    L.append('## 三、OOS 因子覆盖')
    L.append('')
    cov = oos.groupby(oos['month'].astype(str)).agg(
        n=('ts_code', 'size'),
        f_dir_fnn=('f_dir_fnn', lambda s: s.notna().mean()),
        f_mis_xgb_sn=('f_mis_xgb_sn', lambda s: s.notna().mean()),
        ret_fwd=('ret_fwd', lambda s: s.notna().mean()))
    L.append('| 月份 | 股票数 | 直接FNN覆盖率 | 错配XGB覆盖率 | ret_fwd覆盖 |')
    L.append('|---|---|---|---|---|')
    for i, r in cov.iterrows():
        L.append('| %s | %d | %.1f%% | %.1f%% | %.1f%% |' % (
            i, r['n'], r['f_dir_fnn'] * 100, r['f_mis_xgb_sn'] * 100, r['ret_fwd'] * 100))
    L.append('')
    L.append('## 四、口径说明与限制')
    L.append('')
    L.append('- OOS 训练窗口 = 目标月前 5 个月，逐月重训，无前视。')
    L.append('- 缺失值填充的「全样本中位数」仅用 2025-05 及以前估计，避免 OOS 前视。')
    L.append('- 2026-09 的 ret_fwd 需 2026-10 月收益，尚未完成 → 该月只产出选股，不参与评估。')
    L.append('- 指数为价格指数（不含股息），个股收益为除权口径（含股息）→ 基准略被低估。')
    L.append('- 未计冲击成本；小盘股实际滑点高于 20bp。')
    with open(os.path.join(OUT_DIR, 'longonly_oos_report_20261019.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))
    log('  已写 out_v2/longonly_oos_report_20261019.md')


if __name__ == '__main__':
    main()
