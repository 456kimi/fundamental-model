# -*- coding: utf-8 -*-
"""补齐 2009-2015 全市场日线/市值/复权因子（按 trade_date 批量，一次返回全市场）。

输出: data_mkt/mkt_daily_YYYY.parquet, mkt_basic_YYYY.parquet, mkt_adjfactor_YYYY.parquet
按年落盘, 断点续传（文件存在且非空即跳过）。
"""
import os, sys, io, time, traceback
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import pandas as pd
import tushare as ts

TOKEN = __import__("_secrets").TOKEN  # 凭证外置，见 _secrets.py
BASE = r'H:\Claw_backup\财报分析模型\repro\data_mkt'
LOG = os.path.join(BASE, 'fetch_mkt.log')
START_Y, END_Y = 2009, 2015
SLEEP = 0.35
APIS = ['daily', 'daily_basic', 'adj_factor']

ts.set_token(TOKEN)
pro = ts.pro_api()
os.makedirs(BASE, exist_ok=True)


def log(m):
    line = '%s %s' % (pd.Timestamp.now().strftime('%m-%d %H:%M:%S'), m)
    print(line, flush=True)
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def ypath(y, api):
    return os.path.join(BASE, 'mkt_%s_%d.parquet' % (api, y))


def fetch_day(api, day):
    for a in range(1, 4):
        try:
            df = getattr(pro, api)(trade_date=day)
            return df if df is not None else pd.DataFrame()
        except Exception as ex:
            msg = str(ex)
            wait = 60 if ('每分钟' in msg or 'limit' in msg.lower()) else 10
            log('   %s %s try%d FAIL %s wait%ds' % (day, api, a, msg[:70], wait))
            time.sleep(wait)
    return None


def main():
    cal = pro.trade_cal(exchange='SSE', start_date='%d0101' % START_Y,
                        end_date='%d1231' % END_Y, is_open='1')
    days = sorted(cal['cal_date'].tolist())
    log('===== MKT fetch %d..%d | %d trading days | %s =====' % (START_Y, END_Y, len(days), '/'.join(APIS)))
    t0 = time.time()
    done_days = 0
    fail = 0
    for y in range(START_Y, END_Y + 1):
        yd = [d for d in days if d.startswith(str(y))]
        if not yd:
            continue
        if all(os.path.exists(ypath(y, a)) and os.path.getsize(ypath(y, a)) > 1000 for a in APIS):
            log('[%d] exists, skip' % y)
            continue
        log('[%d] %d days' % (y, len(yd)))
        acc = {a: [] for a in APIS}
        for d in yd:
            for a in APIS:
                df = fetch_day(a, d)
                time.sleep(SLEEP)
                if df is None:
                    fail += 1
                elif len(df):
                    acc[a].append(df)
            done_days += 1
            if done_days % 100 == 0:
                el = time.time() - t0
                log('  %d/%d days | fail %d | %.1fmin | eta %.1fmin' % (
                    done_days, len(days), fail, el / 60,
                    el / done_days * (len(days) - done_days) / 60))
        for a in APIS:
            df = pd.concat(acc[a], ignore_index=True) if acc[a] else pd.DataFrame()
            df.to_parquet(ypath(y, a), index=False)
        log('[%d] saved %s' % (y, {a: (sum(len(x) for x in acc[a]) if acc[a] else 0) for a in APIS}))
    log('===== done %d days | fail %d | %.1f min =====' % (done_days, fail, (time.time() - t0) / 60))


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        log('!! interrupted, rerun to resume')
    except Exception:
        log('!! error:\n' + traceback.format_exc()[-800:])
