# -*- coding: utf-8 -*-
import os, sys, io, glob
import numpy as np
import pandas as pd
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

REPRO = r'h:\Claw_backup\财报分析模型\repro'
EVENTS = os.path.join(REPRO, 'data_ext', 'fin_events_full.parquet')
MKT_DIR = os.path.join(REPRO, 'data_mkt')
MEDIA = os.path.join(REPRO, 'data_ext', 'media_proxy_monthly.parquet')
STOCK_PV = r'h:\Claw_backup\vp_probe_data\stock_pv.parquet'
MV_PQ = r'h:\Claw_backup\etf_data\stock_mv_sector.parquet'
OUT = os.path.join(REPRO, 'data_ext', 'monthly_panel_ext.parquet')

START = pd.Timestamp('2008-06-30')
END = pd.Timestamp('2025-06-30')

FIN_VARS = ['AT', 'SEQ', 'ICAPT', 'PSTK', 'PPENT', 'LT', 'CEQ', 'AO', 'DLTT', 'LO', 'ACO',
            'CHE', 'LCO', 'AP', 'DV', 'SALE', 'XIDO', 'IB', 'NI', 'PI', 'TXT', 'NOPI',
            'DVCASH', 'FINACT', 'PURCHASE', 'RECOVER', 'INVACT', 'EMPLOYEE', 'CASH',
            'GPR', 'ART', 'IT', 'RD', 'CUSTOMER', 'SUPPLIER']


def log(m):
    print(m, flush=True)


def code_to_ts(code):
    c = code.astype(str).str.zfill(6)
    p2 = c.str[:2]
    return c + np.select([p2.isin(['60', '68']), p2.isin(['00', '30'])], ['.SH', '.SZ'], '.BJ')


def build_mkt():
    """统一日频行情：close/pct_chg/amount（旧 2016+ 与新 2009-2015 合并）。"""
    parts = []
    pv = pd.read_parquet(STOCK_PV, columns=['ts_code', 'trade_date', 'close', 'pct_chg', 'amount'])
    pv['trade_date'] = pd.to_datetime(pv['trade_date'])
    parts.append(pv)
    n_new = 0
    for f in sorted(glob.glob(os.path.join(MKT_DIR, 'mkt_daily_[0-9][0-9][0-9][0-9].parquet'))):
        d = pd.read_parquet(f, columns=['ts_code', 'trade_date', 'close', 'pct_chg', 'amount'])
        d['trade_date'] = pd.to_datetime(d['trade_date'])
        parts.append(d)
        n_new += len(d)
    daily = pd.concat(parts, ignore_index=True).drop_duplicates(['ts_code', 'trade_date'])
    log('  日频行情: %d 行 (%s..%s) | 新拉 %d 行' % (
        len(daily), daily.trade_date.min().date(), daily.trade_date.max().date(), n_new))
    daily['month_end'] = daily['trade_date'] + pd.offsets.MonthEnd(0)
    # 月度收益 = 月内日收益复利（关键: 不能用月末单日 pct_chg）
    daily['g'] = 1.0 + daily['pct_chg'].fillna(0) / 100.0
    mret = daily.groupby(['ts_code', 'month_end'])['g'].prod().reset_index()
    mret['mret'] = mret['g'] - 1.0
    mret = mret[['ts_code', 'month_end', 'mret']]
    last = daily.groupby(['ts_code', 'month_end']).tail(1)[['ts_code', 'month_end', 'close']]
    last = last.rename(columns={'close': 'me_close'})
    g = daily.groupby(['ts_code', 'month_end'])
    extra = g.agg(amt_mean=('amount', 'mean'),
                  up_ratio=('pct_chg', lambda s: (s > 0).mean())).reset_index()
    m = last.merge(extra, on=['ts_code', 'month_end']).merge(mret, on=['ts_code', 'month_end'])
    m['NEWS'] = np.log1p(m['amt_mean'].clip(lower=0))
    m['MEDIA'] = m['up_ratio']
    log('  月度收益: 均值 %.4f 标准差 %.4f' % (m['mret'].mean(), m['mret'].std()))
    return m


def build_mv():
    """统一月末总市值（元）：旧 stock_mv_sector + 新 daily_basic(万元×1e4)。"""
    parts = []
    mv = pd.read_parquet(MV_PQ)
    mv['date'] = pd.to_datetime(mv['date'])
    mv['ts_code'] = code_to_ts(mv['code'])
    mv['month_end'] = mv['date'] + pd.offsets.MonthEnd(0)
    a = mv.groupby(['ts_code', 'month_end']).tail(1)[['ts_code', 'month_end', 'total_mv', 'sw_industry']]
    a['mv_src'] = 'sector'
    parts.append(a)
    for f in sorted(glob.glob(os.path.join(MKT_DIR, 'mkt_daily_basic_[0-9][0-9][0-9][0-9].parquet'))):
        b = pd.read_parquet(f, columns=['ts_code', 'trade_date', 'total_mv', 'turnover_rate'])
        b['trade_date'] = pd.to_datetime(b['trade_date'])
        b['month_end'] = b['trade_date'] + pd.offsets.MonthEnd(0)
        c = b.groupby(['ts_code', 'month_end']).tail(1)[['ts_code', 'month_end', 'total_mv', 'turnover_rate']]
        c['total_mv'] = c['total_mv'] * 1e4          # 万元 -> 元
        c['mv_src'] = 'tushare'
        parts.append(c)
    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values(['ts_code', 'month_end', 'mv_src'])
    out = out.drop_duplicates(['ts_code', 'month_end'], keep='first')  # sector 优先(与旧口径一致)
    log('  市值: %d 行 (%s..%s)' % (len(out), out.month_end.min().date(), out.month_end.max().date()))
    return out


def build():
    log('===== 构建月度 PIT 面板 =====')
    ev = pd.read_parquet(EVENTS)
    ev['ann_visible'] = pd.to_datetime(ev['ann_visible'])
    ev['end_date'] = pd.to_datetime(ev['end_date'])
    log('事件表: %d 行 | 股票 %d' % (len(ev), ev['ts_code'].nunique()))

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
    log('as-of 后: %d 行 | 有财报 %.1f%%' % (len(panel), 100 * panel['ann_visible'].notna().mean()))

    panel = panel.merge(build_mkt(), on=['ts_code', 'month_end'], how='inner')
    panel = panel.merge(build_mv(), on=['ts_code', 'month_end'], how='left')
    log('并入行情+市值后: %d 行' % len(panel))

    panel = panel.sort_values(['ts_code', 'month_end'])
    panel['ret_fwd'] = panel.groupby('ts_code')['mret'].shift(-1)
    panel['V'] = panel['total_mv']

    bad = (panel['ann_visible'] > panel['month_end']).sum()
    log('① 无前视检查: ann_visible > month_end 行数 = %d (要求 0) %s' % (bad, 'PASS' if bad == 0 else 'FAIL'))
    cur = panel[panel['month_end'] <= END]
    log('② 面板: %d 行 | 股票 %d | %s..%s' % (
        len(cur), cur['ts_code'].nunique(), cur.month_end.min().date(), cur.month_end.max().date()))
    log('③ 市值来源分布: %s' % cur['mv_src'].value_counts(dropna=False).to_dict())
    cov = cur.groupby(cur['month_end'].dt.year).apply(
        lambda d: pd.Series({'n': len(d), 'V': d['V'].notna().mean(),
                             'NI': d['NI'].notna().mean(), 'ret_fwd': d['ret_fwd'].notna().mean()}),
        include_groups=False)
    log('④ 分年覆盖:\n%s' % cov.round(3).to_string())
    cur.to_parquet(OUT, index=False)
    log('已保存: %s (%.1f MB)' % (OUT, os.path.getsize(OUT) / 1e6))


if __name__ == '__main__':
    build()
