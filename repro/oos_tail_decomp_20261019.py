# -*- coding: utf-8 -*-
"""尾部诊断：把多头超额拆成「beta / 规模 / 行业 / 残差（选股）」。

方法（holdings-based 归因，标准做法）：
  1. 每月对全市场做横截面回归：ret_fwd ~ beta + logV + 行业哑变量（无截距，行业全覆盖）
     → 得到各特征的「特征收益」γ
  2. 组合的特征暴露 X_pf（持仓等权平均），基准 X_bm（全池等权平均）
  3. 贡献 = (X_pf − X_bm) · γ
  4. 残差 = 多头超额 − Σ贡献   ← 这才是真正的选股 alpha

beta：用 t 月之前 24 个月的月度收益对等权市场回归（严格只用历史，shift 1）。
输出：out_v2/tail_decomp_report.md + factors/tail_decomp.csv + 图
"""
import io
import os
import sys
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

REPRO = r'h:\Claw_backup\财报分析模型\repro'
OUT = os.path.join(REPRO, 'out_v2')
FACTOR_DIR = os.path.join(REPRO, 'factors')
IS_PANEL = os.path.join(REPRO, 'data_ext', 'monthly_panel_ext.parquet')
OOS_PANEL = os.path.join(REPRO, 'data_oos', 'monthly_panel_oos.parquet')
IS_FACTOR = os.path.join(FACTOR_DIR, 'funda_ml_factor_panel.parquet')
OOS_FACTOR = os.path.join(OUT, 'oos_factor_panel.parquet')

OOS_START = pd.Period('2025-06', 'M')
OOS_END = pd.Period('2026-08', 'M')
BETA_WIN = 24
MIN_OBS = 18
COST = 0.0020
ENTRY_PCT, HOLD_PCT = 0.10, 0.25


def log(m=''):
    print(m, flush=True)


def sec(t):
    log('\n' + '=' * 80)
    log('### ' + t)
    log('=' * 80)


# ---------------------------------------------------------------- 数据
def load_panel():
    a = pd.read_parquet(IS_PANEL, columns=['ts_code', 'month_end', 'mret', 'V', 'sw_industry', 'ret_fwd'])
    b = pd.read_parquet(OOS_PANEL, columns=['ts_code', 'month_end', 'mret', 'V', 'sw_industry', 'ret_fwd'])
    for d in (a, b):
        d['month_end'] = pd.to_datetime(d['month_end'])
    a = a[a['month_end'] < '2025-06-30']
    p = pd.concat([a, b], ignore_index=True)
    p = p.drop_duplicates(['ts_code', 'month_end'])
    p['month'] = p['month_end'].dt.to_period('M')
    p = p[p['ts_code'].str.endswith(('.SH', '.SZ'))]
    p = p[~p['ts_code'].str.startswith('688')]
    p = p[p['V'].notna() & (p['V'] > 0)]
    p['lV'] = np.log(p['V'])
    return p


def calc_beta(p):
    """滚动 24 个月 beta（只用 t 之前的数据）。"""
    wide = p.pivot_table(index='month', columns='ts_code', values='mret')
    mkt = wide.mean(axis=1)                       # 等权市场
    var = mkt.rolling(BETA_WIN, min_periods=MIN_OBS).var()
    cov = wide.rolling(BETA_WIN, min_periods=MIN_OBS).cov(mkt)
    beta = cov.div(var, axis=0).shift(1)          # t 月用 t-1 及以前
    cnt = wide.rolling(BETA_WIN, min_periods=1).count().shift(1)
    beta = beta.where(cnt >= MIN_OBS)
    long = beta.stack().rename('beta').reset_index()
    long.columns = ['month', 'ts_code', 'beta']
    log('  beta 估计: %d 个(月,股票) | 覆盖 %.1f%% | 中位 %.3f'
        % (len(long), 100 * beta.notna().sum().sum() / max(wide.notna().sum().sum(), 1),
           long['beta'].median()))
    return long, mkt


# ---------------------------------------------------------------- 组合
def holdings(fac, col, months):
    prev, out = None, {}
    for mt in months:
        g = fac[fac['month'] == mt].dropna(subset=[col])
        if len(g) < 50:
            continue
        g = g.copy()
        g['rk'] = g[col].rank(method='first')
        n_long = max(int(round(len(g) * ENTRY_PCT)), 5)
        n_hold = max(int(round(len(g) * HOLD_PCT)), n_long)
        top = list(g.sort_values('rk', ascending=False)['ts_code'])
        pool = set(top[:n_hold])
        hold = set(top[:n_long])
        if prev is not None:
            keep = prev & pool
            add = [c for c in top if c not in keep][:max(n_long - len(keep), 0)]
            hold = set(list(keep) + add)
        out[mt] = hold
        prev = hold
    return out


# ---------------------------------------------------------------- 归因
def xs_regress(g, ind_cols):
    """ret_fwd ~ beta + lV + 行业哑变量（无截距）。返回系数 dict 与设计矩阵列名。"""
    X = np.column_stack([g['beta'].values, g['lV'].values, g[ind_cols].values])
    y = g['ret_fwd'].values
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    if ok.sum() < 200:
        return None
    X, y = X[ok], y[ok]
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef


def main():
    sec('0. 载入')
    p = load_panel()
    log('  面板 %d 行 | %s..%s | 股票 %d'
        % (len(p), p['month'].min(), p['month'].max(), p['ts_code'].nunique()))

    beta, mkt = calc_beta(p)
    p = p.merge(beta, on=['month', 'ts_code'], how='left')
    p['beta'] = p['beta'].fillna(p.groupby('month')['beta'].transform('median'))
    p['lV'] = p['lV'].fillna(p.groupby('month')['lV'].transform('median'))
    p['ind'] = p['sw_industry'].fillna('未知')

    # 因子
    isf = pd.read_parquet(IS_FACTOR, columns=['ts_code', 'month_end', 'f_dir_fnn'])
    isf['month'] = pd.to_datetime(isf['month_end']).dt.to_period('M')
    o = pd.read_parquet(OOS_FACTOR, columns=['ts_code', 'month', 'f_dir_fnn'])
    fac = pd.concat([isf[['ts_code', 'month', 'f_dir_fnn']], o], ignore_index=True)
    fac = fac.sort_values(['ts_code', 'month'])
    fac['f_s3'] = fac.groupby('ts_code')['f_dir_fnn'].transform(
        lambda s: s.rolling(3, min_periods=1).mean())

    # 形态：需要 ret_fwd 才能归因
    ev = p.dropna(subset=['ret_fwd']).copy()
    inds = sorted(ev['ind'].unique())
    ind_dum = pd.get_dummies(ev['ind'], prefix='i').astype(float)
    ev = pd.concat([ev, ind_dum], axis=1)
    ind_cols = list(ind_dum.columns)
    log('  行业数 %d | 可用于归因的行 %d' % (len(ind_cols), len(ev)))

    periods = [('样本内 2016-25', ev[(ev['month'] >= pd.Period('2016-01', 'M'))]),
               ('样本外 2025-06~2026-08',
                ev[(ev['month'] >= OOS_START) & (ev['month'] <= OOS_END)])]

    rows = []
    detail = {}
    for pname, sub in periods:
        months = sorted(sub['month'].unique())
        hs = holdings(fac, 'f_s3', months)
        recs = []
        for mt in months:
            g = sub[sub['month'] == mt]
            hold = hs.get(mt)
            if not hold or len(g) < 200:
                continue
            coef = xs_regress(g, ind_cols)
            if coef is None:
                continue
            gamma = dict(zip(['beta', 'lV'] + ind_cols, coef))
            ish = g['ts_code'].isin(hold)
            if ish.sum() < 30:
                continue
            # 暴露
            xb_pf = g.loc[ish, 'beta'].mean()
            xb_bm = g['beta'].mean()
            xs_pf = g.loc[ish, 'lV'].mean()
            xs_bm = g['lV'].mean()
            w_pf = g.loc[ish, ind_cols].mean()
            w_bm = g[ind_cols].mean()
            c_beta = (xb_pf - xb_bm) * gamma['beta']
            c_size = (xs_pf - xs_bm) * gamma['lV']
            c_ind = float(((w_pf - w_bm) * pd.Series(gamma)[ind_cols]).sum())
            ex = g.loc[ish, 'ret_fwd'].mean() - g['ret_fwd'].mean()
            recs.append({'month': mt, 'excess': ex, 'c_beta': c_beta,
                         'c_size': c_size, 'c_ind': c_ind,
                         'resid': ex - c_beta - c_size - c_ind,
                         'beta_pf': xb_pf, 'beta_bm': xb_bm,
                         'lV_pf': xs_pf, 'lV_bm': xs_bm})
        df = pd.DataFrame(recs)
        detail[pname] = df
        if df.empty:
            continue
        log('\n--- %s（%d 个月）---' % (pname, len(df)))
        log('  组合 beta %.3f vs 基准 %.3f | 组合 ln市值 %.2f vs 基准 %.2f'
            % (df['beta_pf'].mean(), df['beta_bm'].mean(),
               df['lV_pf'].mean(), df['lV_bm'].mean()))
        log('  %-12s %9s %9s %9s %9s %9s' % ('口径', '超额', 'beta', '规模', '行业', '残差(选股)'))
        log('  %-12s %+9.3f %+9.3f %+9.3f %+9.3f %+9.3f'
            % ('月均 bp', df['excess'].mean() * 1e4, df['c_beta'].mean() * 1e4,
               df['c_size'].mean() * 1e4, df['c_ind'].mean() * 1e4, df['resid'].mean() * 1e4))
        log('  %-12s %+9.3f %+9.3f %+9.3f %+9.3f %+9.3f'
            % ('年化 %', ((1 + df['excess']).prod() ** (12 / len(df)) - 1) * 100,
               df['c_beta'].mean() * 12 * 100, df['c_size'].mean() * 12 * 100,
               df['c_ind'].mean() * 12 * 100, df['resid'].mean() * 12 * 100))
        rows.append({'区间': pname, '月数': len(df),
                     '超额_bp': df['excess'].mean() * 1e4,
                     'beta_bp': df['c_beta'].mean() * 1e4,
                     '规模_bp': df['c_size'].mean() * 1e4,
                     '行业_bp': df['c_ind'].mean() * 1e4,
                     '残差_bp': df['resid'].mean() * 1e4,
                     'beta_pf': df['beta_pf'].mean()})

    sec('1. 急跌月拆解（2026-06 建仓 → 2026-07 兑现）')
    oo = detail.get('样本外 2025-06~2026-08')
    if oo is not None and not oo.empty:
        w = oo.loc[oo['excess'].idxmin()]
        log('  月份 %s' % w['month'])
        log('  超额 %.2f%% = beta %+.2f%% + 规模 %+.2f%% + 行业 %+.2f%% + 残差 %+.2f%%'
            % (w['excess'] * 100, w['c_beta'] * 100, w['c_size'] * 100,
               w['c_ind'] * 100, w['resid'] * 100))
        log('  该月组合 beta %.3f（基准 %.3f）| 组合 ln市值 %.2f（基准 %.2f）'
            % (w['beta_pf'], w['beta_bm'], w['lV_pf'], w['lV_bm']))
        log('\n  样本外逐月（%%）：')
        log('  %-9s %8s %8s %8s %8s %8s %7s' % (
            '月份', '超额', 'beta', '规模', '行业', '残差', '组合beta'))
        for _, r in oo.iterrows():
            log('  %-9s %+8.2f %+8.2f %+8.2f %+8.2f %+8.2f %7.2f' % (
                r['month'], r['excess'] * 100, r['c_beta'] * 100, r['c_size'] * 100,
                r['c_ind'] * 100, r['resid'] * 100, r['beta_pf']))

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(FACTOR_DIR, 'tail_decomp.csv'), index=False, encoding='utf-8-sig')
    if oo is not None:
        oo.to_csv(os.path.join(FACTOR_DIR, 'tail_decomp_oos_monthly.csv'),
                  index=False, encoding='utf-8-sig')

    sec('2. 结论性判断')
    if out is not None and len(out):
        for _, r in out.iterrows():
            share = (r['beta_bp'] + r['规模_bp'] + r['行业_bp']) / r['超额_bp'] * 100 \
                if r['超额_bp'] else np.nan
            log('  %-22s 超额 %+7.2f bp/月 | 风格解释 %+7.2f bp（%.0f%%）| 选股残差 %+7.2f bp | beta %.2f'
                % (r['区间'], r['超额_bp'], r['beta_bp'] + r['规模_bp'] + r['行业_bp'],
                   share, r['残差_bp'], r['beta_pf']))
    log('\n完成。')


if __name__ == '__main__':
    main()
