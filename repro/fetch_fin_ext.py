# -*- coding: utf-8 -*-
import os, sys, io, time, traceback
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import pandas as pd
import tushare as ts

TOKEN = __import__("_secrets").TOKEN  # 凭证外置，见 _secrets.py
BASE = r'H:\Claw_backup\财报分析模型\repro\data_ext'
BY_BATCH = os.path.join(BASE, 'by_batch')
PILOT = os.path.join(BASE, 'pilot')
LOG = os.path.join(BASE, 'fetch.log')
SRC_LIST = r'H:\Claw_backup\fin_quant\data\stock_list.csv'

START, END = '20080101', '20261231'
BATCH = 300
SLEEP = 0.40

APIS = {
    'income':         dict(fields=None, dated=True),
    'balancesheet':   dict(fields=None, dated=True),
    'cashflow':       dict(fields=None, dated=True),
    'fina_indicator': dict(fields=None, dated=True),
    'top10_holders':  dict(fields='ts_code,ann_date,end_date,holder_name,hold_ratio,holder_type', dated=True),
    'pledge_stat':    dict(fields='ts_code,end_date,pledge_count,pledge_ratio', dated=False),
    'share_float':    dict(fields='ts_code,ann_date,float_date,float_share,float_ratio', dated=True),
}

ts.set_token(TOKEN)
pro = ts.pro_api()
os.makedirs(BY_BATCH, exist_ok=True)
os.makedirs(PILOT, exist_ok=True)


def log(m):
    line = '%s %s' % (pd.Timestamp.now().strftime('%m-%d %H:%M:%S'), m)
    print(line, flush=True)
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def stock_list():
    df = pd.read_csv(SRC_LIST, dtype=str)
    return df['ts_code'].dropna().drop_duplicates().tolist()


def bpath(outdir, i, api):
    return os.path.join(outdir, 'batch_%04d_%s.parquet' % (i, api))


def batch_done(outdir, i):
    return all(os.path.exists(bpath(outdir, i, a)) for a in APIS)


PAGE = 100
PAGINATED = {'balancesheet', 'cashflow', 'fina_indicator'}


def _one(api, code, offset):
    cfg = APIS[api]
    kw = {'ts_code': code}
    if cfg['dated']:
        kw['start_date'] = START
        kw['end_date'] = END
    if cfg['fields']:
        kw['fields'] = cfg['fields']
    if offset:
        kw['offset'] = offset
    for attempt in range(1, 4):
        try:
            df = getattr(pro, api)(**kw)
            return df if df is not None else pd.DataFrame()
        except Exception as ex:
            msg = str(ex)
            wait = 60 if ('每分钟' in msg or 'limit' in msg.lower()) else 12
            log('    %s %s off%d try%d FAIL %s wait%ds' % (code, api, offset, attempt, msg[:70], wait))
            time.sleep(wait)
    return None


def call(api, code):
    frames = []
    off = 0
    while True:
        df = _one(api, code, off)
        if df is None:
            return None
        frames.append(df)
        if api not in PAGINATED or len(df) < PAGE or off >= 500:
            break
        off += PAGE
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main():
    pilot = '--pilot' in sys.argv
    outdir = PILOT if pilot else BY_BATCH
    codes = stock_list()
    if pilot:
        codes = codes[:100]
    n = len(codes)
    nb = (n + BATCH - 1) // BATCH
    log('===== EXT fetch %s | %d stocks | %d batches | %d apis | %s..%s =====' % (
        '[PILOT]' if pilot else '', n, nb, len(APIS), START, END))
    failed = []
    t0 = time.time()
    done = 0
    for bi in range(1, nb + 1):
        if batch_done(outdir, bi):
            log('[batch %d/%d] exists, skip' % (bi, nb))
            continue
        lo, hi = (bi - 1) * BATCH, min(bi * BATCH, n)
        log('[batch %d/%d] stocks %d..%d' % (bi, nb, lo + 1, hi))
        acc = {a: [] for a in APIS}
        for c in codes[lo:hi]:
            for a in APIS:
                df = call(a, c)
                time.sleep(SLEEP)
                if df is None:
                    failed.append((c, a, 'fail3'))
                elif len(df):
                    acc[a].append(df)
            done += 1
            if done % 50 == 0:
                el = time.time() - t0
                log('  %d/%d | fail %d | %.0fmin | eta %.0fmin' % (
                    done, n, len(failed), el / 60, el / done * (n - done) / 60))
        for a in APIS:
            df = pd.concat(acc[a], ignore_index=True) if acc[a] else pd.DataFrame()
            df.to_parquet(bpath(outdir, bi, a), index=False)
        rows = {a: (sum(len(x) for x in acc[a]) if acc[a] else 0) for a in APIS}
        log('[batch %d/%d] saved %s' % (bi, nb, rows))
    pd.DataFrame(failed, columns=['ts_code', 'api', 'err']).to_csv(
        os.path.join(outdir, 'failed.csv'), index=False, encoding='utf-8-sig')
    log('===== done %d stocks | fail %d | %.0f min =====' % (done, len(failed), (time.time() - t0) / 60))


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        log('!! interrupted, rerun to resume')
    except Exception:
        log('!! error:\n' + traceback.format_exc()[-800:])
