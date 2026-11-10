# -*- coding: utf-8 -*-
"""降换手改造：信号平滑 / 持仓缓冲带 / 降频调仓，评估换手-收益权衡。

变体：
  A 基线        每月调仓，无平滑无缓冲
  B 平滑3M      因子取 3 个月移动平均后排序
  C 缓冲带      持有至跌出前25%（多头）/ 升出后75%（空头）
  D 平滑+缓冲    B + C
  E 双月调仓    每 2 个月才调仓（中间月持有）

输出：factors/turnover_optimize.csv, out_v2/turnover_optimize_report.md
"""
import io, sys, os
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

FACTOR_DIR = r'h:\Claw_backup\财报分析模型\repro\factors'
OUT_DIR = r'h:\Claw_backup\财报分析模型\repro\out_v2'
PQ = os.path.join(FACTOR_DIR, 'funda_ml_factor_panel.parquet')

FACTORS = {
    'f_dir_gbdt': '直接收益-GBDT',
    'f_mis_fnn_sn': '错配FNN(规模中性)',
    'f_dir_fnn': '直接收益-FNN',
    'f_mis_xgb_sn': '错配XGB(规模中性)',
}
COSTS = [0.0010, 0.0020, 0.0030]
ENTRY_PCT, HOLD_PCT = 0.10, 0.25      # 建仓前10%，持有至跌出前25%


def load():
    df = pd.read_parquet(PQ)
    df['month'] = pd.to_datetime(df['month_end']).dt.to_period('M')
    df['year'] = df['month'].astype(str).str[:4].astype(int)
    df = df.sort_values(['ts_code', 'month'])
    # 信号平滑：个股 3 个月移动平均
    for c in FACTORS:
        df[c + '_s3'] = (df.groupby('ts_code')[c]
                           .transform(lambda s: s.rolling(3, min_periods=1).mean()))
    return df


def pct_rank(g, col):
    return g[col].rank(pct=True, method='first')


def run_variant(df, col, mode):
    """返回 (毛收益序列, 换手序列)。"""
    months = sorted(df['month'].unique())
    prev_l, prev_s = None, None
    gross, tous = [], []
    for i, mt in enumerate(months):
        g = df[df['month'] == mt].dropna(subset=[col, 'ret_fwd']).copy()
        if len(g) < 50:
            continue
        g['pr'] = pct_rank(g, col)
        # 是否调仓
        rebal = True
        if mode == 'E' and i % 2 == 1:
            rebal = False
        if not rebal and prev_l is not None:
            long_m, short_m = prev_l, prev_s
        else:
            n_long = max(int(round(len(g) * ENTRY_PCT)), 5)
            n_hold = max(int(round(len(g) * HOLD_PCT)), n_long)
            # pr 大 = 因子值高 = 预期收益高 → 多头取 nlargest，空头取 nsmallest
            top_names = set(g.nlargest(n_long, 'pr')['ts_code'])
            hold_names = set(g.nlargest(n_hold, 'pr')['ts_code'])
            bot_names = set(g.nsmallest(n_long, 'pr')['ts_code'])
            bot_hold = set(g.nsmallest(n_hold, 'pr')['ts_code'])
            if mode in ('C', 'D') and prev_l is not None:
                keep = prev_l & hold_names
                add_pool = [c for c in g.nlargest(n_hold, 'pr')['ts_code'] if c not in keep]
                need = n_long - len(keep)
                long_m = set(list(keep) + add_pool[:max(need, 0)])
                keep_s = prev_s & bot_hold
                add_s = [c for c in g.nsmallest(n_hold, 'pr')['ts_code'] if c not in keep_s]
                need_s = n_long - len(keep_s)
                short_m = set(list(keep_s) + add_s[:max(need_s, 0)])
            else:
                long_m, short_m = top_names, bot_names
        rl = g.loc[g['ts_code'].isin(long_m), 'ret_fwd'].mean()
        rs = g.loc[g['ts_code'].isin(short_m), 'ret_fwd'].mean()
        if np.isnan(rl) or np.isnan(rs):
            continue
        gross.append(rl - rs)
        if prev_l is not None and rebal:
            to_l = 1 - len(prev_l & long_m) / max(len(prev_l), 1)
            to_s = 1 - len(prev_s & short_m) / max(len(prev_s), 1)
            tous.append((to_l, to_s))
        else:
            tous.append((0.0, 0.0))
        prev_l, prev_s = long_m, short_m
    return pd.Series(gross), tous


def summarize(gross, tous, cost):
    g = gross.dropna()
    if len(g) < 2 or g.std() == 0:
        return np.nan, np.nan, np.nan
    to = np.array(tous)
    c = (to[:, 0] + to[:, 1]) * cost * 2
    net = g.values - c
    t = net.mean() / (net.std() / np.sqrt(len(net))) if net.std() else np.nan
    return g.mean() * 100, net.mean() * 100, t


def main():
    df = load()
    print('面板 %d 行 | %d 月 | %d 股' % (len(df), df['month'].nunique(), df['ts_code'].nunique()))

    MODES = [('A', '基线'), ('B', '平滑3M'), ('C', '缓冲带'), ('D', '平滑+缓冲'), ('E', '双月调仓')]
    rows = []
    for period, sub in [('全期', df), ('2016-25', df[df.year >= 2016])]:
        for key, label in FACTORS.items():
            if key not in sub.columns:
                continue
            for mk, mlabel in MODES:
                col = key + '_s3' if mk in ('B', 'D') else key
                gross, tous = run_variant(sub, col, mk)
                toarr = np.array(tous)
                to_avg = (toarr[:, 0].mean() + toarr[:, 1].mean()) / 2 if len(toarr) else np.nan
                rec = {'区间': period, '因子': label, '变体': mlabel,
                       '月换手%': round(to_avg * 100, 1), '年化换手%': round(to_avg * 12 * 100, 0),
                       '毛收益%': round(gross.mean() * 100, 3)}
                for c in COSTS:
                    _, n, t = summarize(gross, tous, c)
                    rec['净_%dbp' % (c * 10000)] = round(n, 3)
                    if abs(c - 0.0020) < 1e-9:
                        rec['净t_20bp'] = round(t, 2)
                rows.append(rec)

    tt = pd.DataFrame(rows)
    tt.to_csv(os.path.join(FACTOR_DIR, 'turnover_optimize.csv'), index=False, encoding='utf-8-sig')

    print()
    print('=' * 120)
    print('【2016-25 降换手改造】')
    print('=' * 120)
    s = tt[tt['区间'] == '2016-25']
    print('%-20s %-10s %9s %10s %9s %9s %9s %8s' % (
        '因子', '变体', '月换手%', '年化换手%', '毛收益%', '净@10bp', '净@20bp', '净t_20bp'))
    for _, r in s.iterrows():
        print('%-20s %-10s %9.1f %10.0f %9.3f %9.3f %9.3f %8.2f' % (
            r['因子'], r['变体'], r['月换手%'], r['年化换手%'], r['毛收益%'],
            r['净_10bp'], r['净_20bp'], r['净t_20bp']))

    print()
    print('=' * 120)
    print('【全期降换手改造】')
    print('=' * 120)
    s = tt[tt['区间'] == '全期']
    print('%-20s %-10s %9s %10s %9s %9s %9s %8s' % (
        '因子', '变体', '月换手%', '年化换手%', '毛收益%', '净@10bp', '净@20bp', '净t_20bp'))
    for _, r in s.iterrows():
        print('%-20s %-10s %9.1f %10.0f %9.3f %9.3f %9.3f %8.2f' % (
            r['因子'], r['变体'], r['月换手%'], r['年化换手%'], r['毛收益%'],
            r['净_10bp'], r['净_20bp'], r['净t_20bp']))

    # 报告
    lines = ['# 降换手改造（平滑 / 缓冲带 / 降频）', '',
             '样本 %d 行 / %d 月 / %d 股' % (len(df), df['month'].nunique(), df['ts_code'].nunique()),
             '', '口径：建仓前 10%，持有至跌出前 25%；平滑=个股因子 3 月移动平均；双月=每 2 月调仓',
             '', '## 2016-25', '',
             '| 因子 | 变体 | 月换手% | 年化换手% | 毛收益% | 净@10bp | 净@20bp | 净t(20bp) |',
             '|---|---|---|---|---|---|---|---|']
    for _, r in tt[tt['区间'] == '2016-25'].iterrows():
        lines.append('| %s | %s | %.1f | %.0f | %.3f | %.3f | %.3f | %.2f |' % (
            r['因子'], r['变体'], r['月换手%'], r['年化换手%'], r['毛收益%'],
            r['净_10bp'], r['净_20bp'], r['净t_20bp']))
    lines += ['', '## 全期', '',
              '| 因子 | 变体 | 月换手% | 年化换手% | 毛收益% | 净@10bp | 净@20bp | 净t(20bp) |',
              '|---|---|---|---|---|---|---|---|']
    for _, r in tt[tt['区间'] == '全期'].iterrows():
        lines.append('| %s | %s | %.1f | %.0f | %.3f | %.3f | %.3f | %.2f |' % (
            r['因子'], r['变体'], r['月换手%'], r['年化换手%'], r['毛收益%'],
            r['净_10bp'], r['净_20bp'], r['净t_20bp']))
    with open(os.path.join(OUT_DIR, 'turnover_optimize_report.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print()
    print('已写 factors/turnover_optimize.csv, out_v2/turnover_optimize_report.md')


if __name__ == '__main__':
    main()
