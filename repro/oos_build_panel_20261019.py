# -*- coding: utf-8 -*-
"""构建 OOS 扩展面板（2025-06 ~ 2026-09），口径严格复刻 build_monthly_panel.py。

数据源：
  财报      data_ext/fin_events_full.parquet（ann_visible 至 2026-09-30）
  日线      data_oos/daily_YYYYMM.parquet（新拉）
  月末市值  data_oos/basic_monthend.parquet（新拉，total_mv 单位=万元 -> ×1e4）
            + etf_data/stock_mv_sector.parquet（sector 优先，与旧面板一致）
  行业      stock_mv_sector.sw_industry

输出：data_oos/monthly_panel_oos.parquet
另做三项一致性校验：
  ① 无前视：ann_visible <= month_end 必须 0 违规
  ② 接缝：新算的 2025-06 月度收益 vs 旧面板 2025-05 的 ret_fwd
  ③ 市值来源：sector 与 tushare 在重叠月末是否同口径
"""
import io
import os
import sys
import glob
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

REPRO = r'h:\Claw_backup\财报分析模型\repro'
OOS = os.path.join(REPRO, 'data_oos')
EVENTS = os.path.join(REPRO, 'data_ext', 'fin_events_full.parquet')
OLD_PANEL = os.path.join(REPRO, 'data_ext', 'monthly_panel_ext.parquet')
MV_PQ = r'h:\Claw_backup\etf_data\stock_mv_sector.parquet'
OUT = os.path.join(OOS, 'monthly_panel_oos.parquet')

FIN_VARS = ['AT', 'SEQ', 'ICAPT', 'PSTK', 'PPENT', 'LT', 'CEQ', 'AO', 'DLTT', 'LO', 'ACO',
            'CHE', 'LCO', 'AP', 'DV', 'SALE', 'XIDO', 'IB', 'NI', 'PI', 'TXT', 'NOPI',
            'DVCASH', 'FINACT', 'PURCHASE', 'RECOVER', 'INVACT', 'EMPLOYEE', 'CASH',
            'GPR', 'ART', 'IT', 'RD', 'CUSTOMER', 'SUPPLIER']

START = pd.Timestamp('2025-06-30')
END = pd.Timestamp('2026-09-30')


def log(m):
    print(m, flush=True)


def build_daily():
    parts = []
    for f in sorted(glob.glob(os.path.join(OOS, 'daily_[0-9][0-9][0-9][0-9][0-9][0-9].parquet'))):
        d = pd.read_parquet(f, columns=['ts_code', 'trade_date', 'close', 'pct_chg', 'amount'])
        d['trade_date'] = pd.to_datetime(d['trade_date'])
        parts.append(d)
    daily = pd.concat(parts, ignore_index=True).drop_duplicates(['ts_code', 'trade_date'])
    log('  日线: %d 行 | %s..%s | 股票 %d' % (
        len(daily), daily.trade_date.min().date(), daily.trade_date.max().date(),
        daily['ts_code'].nunique()))
    return daily


def build_mkt(daily):
    d = daily.copy()
    d['month_end'] = d['trade_date'] + pd.offsets.MonthEnd(0)
    d['g'] = 1.0 + d['pct_chg'].fillna(0) / 100.0
    mret = d.groupby(['ts_code', 'month_end'])['g'].prod().reset_index()
    mret['mret'] = mret['g'] - 1.0
    g = d.groupby(['ts_code', 'month_end'])
    extra = g.agg(amt_mean=('amount', 'mean'),
                  up_ratio=('pct_chg', lambda s: (s > 0).mean())).reset_index()
    m = extra.merge(mret[['ts_code', 'month_end', 'mret']], on=['ts_code', 'month_end'])
    m['NEWS'] = np.log1p(m['amt_mean'].clip(lower=0))
    m['MEDIA'] = m['up_ratio']
    log('  月度收益: %d 行 | 均值 %.4f 标准差 %.4f' % (len(m), m['mret'].mean(), m['mret'].std()))
    return m[['ts_code', 'month_end', 'mret', 'NEWS', 'MEDIA']]


def code_to_ts(code):
    c = code.astype(str).str.zfill(6)
    p2 = c.str[:2]
    return c + np.select([p2.isin(['60', '68']), p2.isin(['00', '30'])], ['.SH', '.SZ'], '.BJ')


def build_mv(true_me):
    """sector 优先，但只接受「date 恰为当月最后一个交易日」的行。
    stock_mv_sector 止于 2026-09-04，若整月不完整则该月整体回落 tushare。
    true_me: {Period: Timestamp} 真实月末交易日
    """
    mv = pd.read_parquet(MV_PQ)
    mv['date'] = pd.to_datetime(mv['date'])
    mv['ts_code'] = code_to_ts(mv['code'])
    mv['month'] = mv['date'].dt.to_period('M')
    mv = mv[mv['month'] >= pd.Period('2025-05', 'M')].copy()
    mv['is_true_me'] = [d == true_me.get(m, pd.NaT) for d, m in zip(mv['date'], mv['month'])]
    a = mv[mv['is_true_me']][['ts_code', 'date', 'total_mv', 'sw_industry']].copy()
    a = a.rename(columns={'date': 'month_end'})
    a['mv_src'] = 'sector'
    log('  sector 真实月末行: %d | 覆盖月份 %d' % (len(a), a['month_end'].dt.to_period('M').nunique()))

    b = pd.read_parquet(os.path.join(OOS, 'basic_monthend.parquet'),
                        columns=['ts_code', 'trade_date', 'total_mv'])
    b['trade_date'] = pd.to_datetime(b['trade_date'])
    b['month'] = b['trade_date'].dt.to_period('M')
    b = b[[d == true_me.get(m, pd.NaT) for d, m in zip(b['trade_date'], b['month'])]].copy()
    b['total_mv'] = b['total_mv'] * 1e4
    b = b.rename(columns={'trade_date': 'month_end'})[['ts_code', 'month_end', 'total_mv']]
    b['mv_src'] = 'tushare'

    # 同口径校验：两个来源在重叠 (ts_code, month_end) 上应一致
    jj = a[['ts_code', 'month_end', 'total_mv']].merge(
        b[['ts_code', 'month_end', 'total_mv']], on=['ts_code', 'month_end'],
        suffixes=('_s', '_t'))
    if len(jj) > 100:
        r = (jj['total_mv_t'] / jj['total_mv_s'])
        log('  [校验③] 重叠 %d 个(股票,月末) | tushare/sector 比值 中位 %.4f  p5 %.4f  p95 %.4f'
            % (len(jj), r.median(), r.quantile(.05), r.quantile(.95)))
    else:
        log('  [校验③] 两源重叠 %d 个，不足比对' % len(jj))

    out = pd.concat([a[['ts_code', 'month_end', 'total_mv', 'sw_industry', 'mv_src']],
                     b[['ts_code', 'month_end', 'total_mv', 'mv_src']]], ignore_index=True)
    # 关键：面板的 month_end 是「日历月末」，这里必须归一化，否则周末月份对不上
    out['month_end'] = out['month_end'] + pd.offsets.MonthEnd(0)
    out = out.sort_values(['ts_code', 'month_end', 'mv_src'])
    out = out.drop_duplicates(['ts_code', 'month_end'], keep='first')
    by = out.groupby(out['month_end'].dt.to_period('M'))['mv_src'].value_counts().unstack(fill_value=0)
    log('  市值来源逐月:\n%s' % by.to_string())
    return out


def check_mv_consistency(mv):
    log('\n  [校验③] 市值来源同口径检查（2026-08 月末，sector vs tushare）')
    s = mv[(mv['mv_src'] == 'sector') & (mv['month_end'] == '2026-08-31')][['ts_code', 'total_mv']]
    t = mv[(mv['mv_src'] == 'tushare') & (mv['month_end'] == '2026-08-31')][['ts_code', 'total_mv']]
    j = s.merge(t, on='ts_code', suffixes=('_s', '_t'))
    if len(j) > 100:
        r = (j['total_mv_t'] / j['total_mv_s']).describe()
        log('    重叠 %d 只 | 比值 中位 %.4f  p5 %.4f  p95 %.4f'
            % (len(j), r['50%'], r['5%'], r['95%']))
    else:
        log('    重叠不足（%d 只），跳过' % len(j))


def main():
    log('===== 构建 OOS 面板 %s .. %s =====' % (START.date(), END.date()))
    ev = pd.read_parquet(EVENTS)
    ev['ann_visible'] = pd.to_datetime(ev['ann_visible'])
    ev['end_date'] = pd.to_datetime(ev['end_date'])

    daily = build_daily()
    mkt = build_mkt(daily)
    true_me = {p: g['trade_date'].max() for p, g in daily.groupby(daily['trade_date'].dt.to_period('M'))}
    log('  真实月末交易日: %s .. %s（共 %d 个月）'
        % (min(true_me.values()).date(), max(true_me.values()).date(), len(true_me)))
    mv = build_mv(true_me)

    months = pd.date_range(START, END, freq='ME')
    left = pd.MultiIndex.from_product([ev['ts_code'].unique(), months],
                                      names=['ts_code', 'month_end']).to_frame(index=False)
    keep = ['ts_code', 'end_date', 'ann_visible', 'report_type'] + FIN_VARS
    e = ev[keep].sort_values(['ts_code', 'ann_visible', 'end_date'])
    e = e.groupby(['ts_code', 'ann_visible'], as_index=False).tail(1)
    panel = pd.merge_asof(left.sort_values(['month_end', 'ts_code']),
                          e.sort_values(['ann_visible', 'ts_code']),
                          left_on='month_end', right_on='ann_visible', by='ts_code',
                          direction='backward')
    log('  as-of 后: %d 行 | 有财报 %.1f%%' % (len(panel), 100 * panel['ann_visible'].notna().mean()))

    panel = panel.merge(mkt, on=['ts_code', 'month_end'], how='inner')
    panel = panel.merge(mv, on=['ts_code', 'month_end'], how='left')
    log('  并入行情+市值: %d 行' % len(panel))

    # 需要 2025-06 的收益做 2025-05 的 ret_fwd 校验，故多留一个月再 shift
    panel = panel.sort_values(['ts_code', 'month_end'])
    panel['ret_fwd'] = panel.groupby('ts_code')['mret'].shift(-1)
    panel['V'] = panel['total_mv']

    bad = (panel['ann_visible'] > panel['month_end']).sum()
    log('\n  [校验①] 无前视: ann_visible > month_end 行数 = %d %s'
        % (bad, 'PASS' if bad == 0 else 'FAIL'))

    log('\n  [校验②] 接缝检查：新算 2025-06 月度收益 vs 旧面板 2025-05 的 ret_fwd')
    old = pd.read_parquet(OLD_PANEL, columns=['ts_code', 'month_end', 'ret_fwd'])
    old['month_end'] = pd.to_datetime(old['month_end'])
    o = old[old['month_end'] == '2025-05-31'][['ts_code', 'ret_fwd']].dropna()
    n = panel[panel['month_end'] == '2025-05-31']
    if len(n) == 0:
        log('    新版无 2025-05 行（START 从 2025-06 起），改用 2025-06 的 mret 对旧 ret_fwd')
        n = panel[panel['month_end'] == '2025-06-30'][['ts_code', 'mret']].rename(
            columns={'mret': 'ret_fwd'}).dropna()
    j = o.merge(n, on='ts_code', suffixes=('_old', '_new'))
    if len(j):
        diff = (j['ret_fwd_new'] - j['ret_fwd_old']).abs()
        log('    重叠 %d 只 | 绝对差 中位 %.6f  p95 %.6f  max %.6f | >1%% 的占比 %.2f%%'
            % (len(j), diff.median(), diff.quantile(0.95), diff.max(),
               100 * (diff > 0.01).mean()))
    else:
        log('    无可比对象')

    cur = panel[panel['month_end'] >= START]
    log('\n  面板: %d 行 | 股票 %d | %s..%s'
        % (len(cur), cur['ts_code'].nunique(), cur.month_end.min().date(), cur.month_end.max().date()))
    log('  市值来源: %s' % cur['mv_src'].value_counts(dropna=False).to_dict())
    cov = cur.groupby(cur['month_end'].dt.to_period('M')).apply(
        lambda d: pd.Series({'n': len(d), 'V': d['V'].notna().mean(),
                             'NI': d['NI'].notna().mean(),
                             'ret_fwd': d['ret_fwd'].notna().mean()}), include_groups=False)
    log('  逐月覆盖:\n%s' % cov.round(3).to_string())

    cur.to_parquet(OUT, index=False)
    log('\n已保存: %s (%.1f MB)' % (OUT, os.path.getsize(OUT) / 1e6))


if __name__ == '__main__':
    main()
