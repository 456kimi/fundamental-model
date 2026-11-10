# -*- coding: utf-8 -*-
"""补齐 OOS 窗口行情：2025-06-01 ~ 2026-09-30 的日线、月末市值、中证1000 指数。

输出到 repro/data_oos/：
  daily_YYYYMM.parquet       (ts_code, trade_date, close, pre_close, pct_chg, high, low, amount)
  basic_monthend.parquet     (月末 trade_date 的 total_mv)
  index_000852.parquet       (中证1000 日线)
断点续传：已存在的月份跳过。
"""
import io
import os
import sys
import time
import traceback
import pandas as pd
import tushare as ts

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

TOKEN = __import__("_secrets").TOKEN  # 凭证外置，见 _secrets.py
BASE = r'H:\Claw_backup\财报分析模型\repro\data_oos'
LOG = os.path.join(BASE, 'fetch_oos.log')
START, END = '20250601', '20260930'
SLEEP = 0.32

ts.set_token(TOKEN)
pro = ts.pro_api()
os.makedirs(BASE, exist_ok=True)


def log(m):
    line = '%s %s' % (pd.Timestamp.now().strftime('%m-%d %H:%M:%S'), m)
    print(line, flush=True)
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def call(api, **kw):
    for a in range(1, 5):
        try:
            df = getattr(pro, api)(**kw)
            return df if df is not None else pd.DataFrame()
        except Exception as ex:
            msg = str(ex)
            wait = 62 if ('每分钟' in msg or 'limit' in msg.lower() or '超过' in msg) else 8 * a
            log('   %s %s try%d FAIL %s wait%ds' % (api, kw, a, msg[:90], wait))
            time.sleep(wait)
    return None


def main():
    cal = call('trade_cal', exchange='SSE', start_date=START, end_date=END, is_open='1')
    days = sorted(cal['cal_date'].tolist())
    log('===== OOS fetch %s..%s | %d trading days =====' % (START, END, len(days)))

    # 1) 日线，按月落盘
    months = sorted({d[:6] for d in days})
    t0 = time.time()
    for ym in months:
        fp = os.path.join(BASE, 'daily_%s.parquet' % ym)
        if os.path.exists(fp) and os.path.getsize(fp) > 1000:
            log('[%s] exists, skip' % ym)
            continue
        acc, fail = [], 0
        for d in [x for x in days if x[:6] == ym]:
            df = call('daily', trade_date=d)
            time.sleep(SLEEP)
            if df is None:
                fail += 1
            elif len(df):
                acc.append(df[['ts_code', 'trade_date', 'open', 'high', 'low', 'close',
                               'pre_close', 'pct_chg', 'amount']])
        out = pd.concat(acc, ignore_index=True) if acc else pd.DataFrame()
        out.to_parquet(fp, index=False)
        log('[%s] %d days -> %d rows | fail %d | %.1fmin'
            % (ym, len([x for x in days if x[:6] == ym]), len(out), fail, (time.time() - t0) / 60))

    # 2) 月末市值
    monthends = []
    for ym in months:
        ds = [d for d in days if d[:6] == ym]
        if ds:
            monthends.append(ds[-1])
    fp = os.path.join(BASE, 'basic_monthend.parquet')
    if os.path.exists(fp) and os.path.getsize(fp) > 1000:
        log('basic_monthend exists, skip')
    else:
        acc = []
        for d in monthends:
            df = call('daily_basic', trade_date=d)
            time.sleep(SLEEP)
            if df is not None and len(df):
                acc.append(df[['ts_code', 'trade_date', 'total_mv', 'turnover_rate']])
        out = pd.concat(acc, ignore_index=True) if acc else pd.DataFrame()
        out.to_parquet(fp, index=False)
        log('basic_monthend: %d month-ends -> %d rows' % (len(monthends), len(out)))

    # 3) 中证1000
    fp = os.path.join(BASE, 'index_000852.parquet')
    if os.path.exists(fp) and os.path.getsize(fp) > 1000:
        log('index exists, skip')
    else:
        idx = call('index_daily', ts_code='000852.SH',
                   start_date=START, end_date=END)
        if idx is not None and len(idx):
            idx.to_parquet(fp, index=False)
            log('index_000852: %d rows %s..%s' % (len(idx), idx['trade_date'].min(), idx['trade_date'].max()))
        else:
            log('index_000852 FAILED')

    log('===== OOS fetch done %.1f min =====' % ((time.time() - t0) / 60))


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        log('!! interrupted')
    except Exception:
        log('!! error:\n' + traceback.format_exc()[-900:])
