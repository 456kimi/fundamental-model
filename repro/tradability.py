# -*- coding: utf-8 -*-
"""可成交性检验：剔除调仓日停牌 / 一字涨跌停后重算因子表现。

判定（月末最后交易日）：
  停牌   = 该股在该日无行情行
  一字板 = high == low（全天单一价格）
  一字涨停 = 一字板 且 pct_chg > 0  → 无法买入
  一字跌停 = 一字板 且 pct_chg < 0  → 无法卖出
口径：
  A 基线     不做可成交性过滤
  B 严格     剔除 停牌 + 一字涨(多) + 一字跌(空)
  C 方向     多头剔一字涨停/停牌；空头剔一字跌停/停牌

输出：factors/tradability.csv, out_v2/tradability_report.md
"""
import io, sys, os, glob
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

REPRO = r'h:\Claw_backup\财报分析模型\repro'
FACTOR_DIR = os.path.join(REPRO, 'factors')
OUT_DIR = os.path.join(REPRO, 'out_v2')
PQ = os.path.join(FACTOR_DIR, 'funda_ml_factor_panel.parquet')
MKT = os.path.join(REPRO, 'data_mkt')
PV = r'h:\Claw_backup\vp_probe_data\stock_pv.parquet'
FLAG_PQ = os.path.join(REPRO, 'data_ext', 'mkt_flags_monthend.parquet')

FACTORS = {
    'f_dir_fnn': '直接收益-FNN',
    'f_dir_gbdt': '直接收益-GBDT',
    'f_mis_xgb_sn': '错配XGB(规模中性)',
    'f_mis_fnn_sn': '错配FNN(规模中性)',
}
COST = 0.0020
ENTRY_PCT, HOLD_PCT = 0.10, 0.25


# ------------------------------------------------------- 1) 构建月末可成交标记
def build_flags():
    if os.path.exists(FLAG_PQ):
        f = pd.read_parquet(FLAG_PQ)
        print('复用可成交标记 %d 行' % len(f))
        return f

    parts = []
    pv = pd.read_parquet(PV, columns=['ts_code', 'trade_date', 'high', 'low', 'pct_chg'])
    pv['trade_date'] = pd.to_datetime(pv['trade_date'])
    parts.append(pv)
    for fn in sorted(glob.glob(os.path.join(MKT, 'mkt_daily_[0-9][0-9][0-9][0-9].parquet'))):
        d = pd.read_parquet(fn, columns=['ts_code', 'trade_date', 'high', 'low', 'pct_chg'])
        d['trade_date'] = pd.to_datetime(d['trade_date'])
        parts.append(d)
    daily = pd.concat(parts, ignore_index=True).drop_duplicates(['ts_code', 'trade_date'])
    print('日频行情 %d 行 | %s..%s' % (len(daily), daily.trade_date.min().date(), daily.trade_date.max().date()))

    # 每月最后交易日（全市场口径）
    alld = pd.Series(daily['trade_date'].unique()).sort_values()
    last_days = set(alld.groupby(alld.dt.to_period('M')).max().tolist())
    me = daily[daily['trade_date'].isin(last_days)].copy()
    me['month'] = me['trade_date'].dt.to_period('M')
    me['is_one_word'] = (me['high'] == me['low'])
    me['one_word_up'] = me['is_one_word'] & (me['pct_chg'] > 0)
    me['one_word_down'] = me['is_one_word'] & (me['pct_chg'] < 0)
    me = me[['ts_code', 'month', 'is_one_word', 'one_word_up', 'one_word_down']]
    me.drop_duplicates(['ts_code', 'month']).to_parquet(FLAG_PQ, index=False)
    print('月末可成交标记 %d 行' % len(me))
    return me


# ------------------------------------------------------- 2) 组合构建（含可成交过滤）
def run(df, col, todable_mode):
    months = sorted(df['month'].unique())
    prev_l = prev_s = None
    gross, tous = [], []
    n_excl = []
    for i, mt in enumerate(months):
        g = df[df['month'] == mt].dropna(subset=[col, 'ret_fwd']).copy()
        if len(g) < 50:
            continue
        n0 = len(g)
        # 可成交过滤
        if todable_mode == 'B':
            g = g[~(g['one_word_up'] | g['one_word_down'] | g['suspended'])]
        elif todable_mode == 'C':
            pass   # 方向性过滤在选股后做
        n_excl.append(1 - len(g) / n0)
        if len(g) < 30:
            continue
        g['rk'] = g[col].rank(method='first')
        n_long = max(int(round(len(g) * ENTRY_PCT)), 5)
        n_hold = max(int(round(len(g) * HOLD_PCT)), n_long)
        up_ok = ~g['one_word_up'].fillna(False).astype(bool)
        dn_ok = ~g['one_word_down'].fillna(False).astype(bool)

        def pick(pool, k, ok):
            cand = g.loc[g['ts_code'].isin(pool) & ok]
            return set(cand.head(k)['ts_code'])

        top = list(g.sort_values('rk', ascending=False)['ts_code'])
        bot = list(g.sort_values('rk')['ts_code'])
        if todable_mode == 'C':
            top = list(g.loc[up_ok].sort_values('rk', ascending=False)['ts_code'])
            bot = list(g.loc[dn_ok].sort_values('rk')['ts_code'])
        pool_l, pool_s = set(top[:n_hold]), set(bot[:n_hold])
        long_m = set(top[:n_long])
        short_m = set(bot[:n_long])
        if prev_l is not None and todable_mode != 'X':
            keep = prev_l & pool_l
            add = [c for c in top if c not in keep][:max(n_long - len(keep), 0)]
            long_m = set(list(keep) + add)
            keep_s = prev_s & pool_s
            add_s = [c for c in bot if c not in keep_s][:max(n_long - len(keep_s), 0)]
            short_m = set(list(keep_s) + add_s)
        rl = g.loc[g['ts_code'].isin(long_m), 'ret_fwd'].mean()
        rs = g.loc[g['ts_code'].isin(short_m), 'ret_fwd'].mean()
        if np.isnan(rl) or np.isnan(rs):
            continue
        gross.append(rl - rs)
        if prev_l is not None:
            to_l = 1 - len(prev_l & long_m) / max(len(prev_l), 1)
            to_s = 1 - len(prev_s & short_m) / max(len(prev_s), 1)
            tous.append((to_l, to_s))
        else:
            tous.append((0.0, 0.0))
        prev_l, prev_s = long_m, short_m
    return pd.Series(gross), tous, (np.mean(n_excl) if n_excl else np.nan)


def summarize(gross, tous, cost=COST):
    g = gross.dropna()
    if len(g) < 2 or g.std() == 0:
        return np.nan, np.nan, np.nan
    to = np.array(tous)
    net = g.values - (to[:, 0] + to[:, 1]) * cost * 2
    t = net.mean() / (net.std() / np.sqrt(len(net))) if net.std() else np.nan
    return g.mean() * 100, net.mean() * 100, t


def main():
    flags = build_flags()
    df = pd.read_parquet(PQ)
    df['month'] = pd.to_datetime(df['month_end']).dt.to_period('M')
    df['year'] = df['month'].astype(str).str[:4].astype(int)
    df = df.merge(flags, on=['ts_code', 'month'], how='left')
    # 无行情行 = 停牌（无法交易）
    df['suspended'] = df['is_one_word'].isna()
    df['one_word_up'] = df['one_word_up'].fillna(False).astype(bool)
    df['one_word_down'] = df['one_word_down'].fillna(False).astype(bool)
    # 信号平滑
    df = df.sort_values(['ts_code', 'month'])
    for c in FACTORS:
        df[c + '_s3'] = df.groupby('ts_code')[c].transform(lambda s: s.rolling(3, min_periods=1).mean())

    print()
    print('停牌(月末无行情)占比: %.2f%%' % (100 * df['suspended'].mean()))
    print('一字涨停占比: %.2f%% | 一字跌停占比: %.2f%%' % (
        100 * df['one_word_up'].fillna(False).mean(), 100 * df['one_word_down'].fillna(False).mean()))

    rows = []
    for period, sub in [('全期', df), ('2016-25', df[df.year >= 2016])]:
        for key, label in FACTORS.items():
            col = key + '_s3'
            for mode, mlabel in [('A', '基线(不过滤)'), ('B', '严格剔除'), ('C', '方向剔除')]:
                gross, tous, ex = run(sub, col, mode)
                g0, n0, t0 = summarize(gross, tous)
                to = np.array(tous)
                toa = (to[:, 0].mean() + to[:, 1].mean()) / 2 if len(to) else np.nan
                rows.append({'区间': period, '因子': label, '口径': mlabel,
                             '剔除占比%': round(ex * 100, 2) if ex == ex else np.nan,
                             '年化换手%': round(toa * 12 * 100, 0),
                             '毛收益%': round(g0, 3), '净@20bp%': round(n0, 3), '净t': round(t0, 2)})
    tt = pd.DataFrame(rows)
    tt.to_csv(os.path.join(FACTOR_DIR, 'tradability.csv'), index=False, encoding='utf-8-sig')

    print()
    print('=' * 118)
    print('【可成交性检验】平滑3M + 缓冲带 + 净@20bp')
    print('=' * 118)
    for period in ['2016-25', '全期']:
        print()
        print('--- %s ---' % period)
        s = tt[tt['区间'] == period]
        print('%-20s %-12s %9s %10s %9s %9s %7s' % (
            '因子', '口径', '剔除%', '年化换手%', '毛收益%', '净@20bp%', '净t'))
        for _, r in s.iterrows():
            print('%-20s %-12s %9.2f %10.0f %9.3f %9.3f %7.2f' % (
                r['因子'], r['口径'], r['剔除占比%'], r['年化换手%'],
                r['毛收益%'], r['净@20bp%'], r['净t']))

    lines = ['# 可成交性检验（剔除调仓日停牌 / 一字涨跌停）', '',
             '- 停牌 = 月末最后交易日无行情行',
             '- 一字板 = 当日 high == low（全天单一价格）',
             '- 口径 B 严格：剔除 停牌 + 一字涨 + 一字跌',
             '- 口径 C 方向：多头剔一字涨停/停牌，空头剔一字跌停/停牌',
             '- 组合：信号平滑3M + 缓冲带（建仓10%/持有25%），净收益按 20bp 单边成本', '']
    for period in ['2016-25', '全期']:
        lines += ['## %s' % period, '',
                  '| 因子 | 口径 | 剔除占比% | 年化换手% | 毛收益% | 净@20bp% | 净t |',
                  '|---|---|---|---|---|---|---|']
        for _, r in tt[tt['区间'] == period].iterrows():
            lines.append('| %s | %s | %.2f | %.0f | %.3f | %.3f | %.2f |' % (
                r['因子'], r['口径'], r['剔除占比%'], r['年化换手%'],
                r['毛收益%'], r['净@20bp%'], r['净t']))
        lines.append('')
    with open(os.path.join(OUT_DIR, 'tradability_report.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print()
    print('已写 factors/tradability.csv, out_v2/tradability_report.md')


if __name__ == '__main__':
    main()
