# -*- coding: utf-8 -*-
"""通用因子体检工具 —— 四项体检：预测力 / 规模中性 / 换手成本 / 可成交性

设计目标：任何因子面板（parquet/csv）传进来，自动补齐辅助数据、跑四项体检、出结论。
辅助数据（收益/市值/行业/可成交标记）缺失时自动从工作区既有数据构建并缓存。

用法示例
--------
# 最简：自动识别列，监测所有数值列
python factor_health_check.py --panel H:\\...\\funda_ml_factor_panel.parquet

# 指定因子列与输出
python factor_health_check.py --panel X.parquet --factors f1,f2 --out my_report

# 因子方向为"值越小越好"时翻转
python factor_health_check.py --panel X.parquet --factors f1 --flip

参数
----
--panel      因子面板路径（必需，parquet 或 csv）
--factors    逗号分隔的因子列；缺省=自动识别数值列（排除元数据列）
--code-col   股票代码列名（缺省自动识别 ts_code/code/symbol）
--date-col   日期列名（缺省自动识别 month_end/date/trade_date/month）
--ret-col    下月收益列名（缺省自动识别；没有则从行情构建）
--mv-col     市值列名（缺省自动识别；没有则补）
--ind-col    行业列名（缺省自动识别；没有则补）
--flip       因子取负（用于"越小越好"的因子）
--start/--end  样本起止（YYYY-MM），可选
--out        输出前缀（默认 report_factor_health）

输出
----
<out>.md    体检报告（总览 + 四项明细 + 结论）
<out>.csv   逐因子指标明细
"""
from __future__ import annotations

import argparse
import glob
import io
import os
import sys
import warnings

warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ---------------------------------------------------------------- 工作区路径
W = r'h:\Claw_backup'
AUX_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_aux')
STOCK_PV = os.path.join(W, 'vp_probe_data', 'stock_pv.parquet')
MKT_DIR = os.path.join(W, '财报分析模型', 'repro', 'data_mkt')
PANEL_EXT = os.path.join(W, '财报分析模型', 'repro', 'data_ext', 'monthly_panel_ext.parquet')
FLAGS = os.path.join(W, '财报分析模型', 'repro', 'data_ext', 'mkt_flags_monthend.parquet')
MV_SECTOR = os.path.join(W, 'etf_data', 'stock_mv_sector.parquet')

Q = 10
COSTS = [0.0010, 0.0020, 0.0030]
CORE_RET_COL = 'ret_fwd'


def log(m=''):
    print(m, flush=True)


# ---------------------------------------------------------------- 辅助数据
def _build_monthly_returns():
    """(ts_code, month, ret_fwd) —— 月内复利 + 下月收益。"""
    parts = []
    if os.path.exists(STOCK_PV):
        d = pd.read_parquet(STOCK_PV, columns=['ts_code', 'trade_date', 'pct_chg'])
        d['trade_date'] = pd.to_datetime(d['trade_date'])
        parts.append(d)
    for f in sorted(glob.glob(os.path.join(MKT_DIR, 'mkt_daily_[0-9][0-9][0-9][0-9].parquet'))):
        d = pd.read_parquet(f, columns=['ts_code', 'trade_date', 'pct_chg'])
        d['trade_date'] = pd.to_datetime(d['trade_date'])
        parts.append(d)
    if not parts:
        return None
    d = pd.concat(parts, ignore_index=True).drop_duplicates(['ts_code', 'trade_date'])
    d['g'] = 1.0 + d['pct_chg'].fillna(0) / 100.0
    d['month'] = d['trade_date'].dt.to_period('M')
    m = d.groupby(['ts_code', 'month'], observed=True)['g'].prod().reset_index()
    m['mret'] = m['g'] - 1.0
    m = m[['ts_code', 'month', 'mret']].sort_values(['ts_code', 'month'])
    m['ret_fwd'] = m.groupby('ts_code')['mret'].shift(-1)
    return m[['ts_code', 'month', 'ret_fwd']]


def _build_mv_ind():
    if os.path.exists(PANEL_EXT):
        d = pd.read_parquet(PANEL_EXT, columns=['ts_code', 'month_end', 'total_mv', 'sw_industry'])
        d['month'] = pd.to_datetime(d['month_end']).dt.to_period('M')
        d = d.drop_duplicates(['ts_code', 'month'])
        return d[['ts_code', 'month', 'total_mv', 'sw_industry']]
    if os.path.exists(MV_SECTOR):
        d = pd.read_parquet(MV_SECTOR)
        d['date'] = pd.to_datetime(d['date'])
        c = d['code'].astype(str).str.zfill(6)
        p2 = c.str[:2]
        d['ts_code'] = c + np.select([p2.isin(['60', '68']), p2.isin(['00', '30'])], ['.SH', '.SZ'], '.BJ')
        d['month'] = d['date'].dt.to_period('M')
        d = d.sort_values(['ts_code', 'month']).drop_duplicates(['ts_code', 'month'], keep='last')
        return d[['ts_code', 'month', 'total_mv', 'sw_industry']]
    return None


def ensure_aux():
    os.makedirs(AUX_DIR, exist_ok=True)
    out = {}
    specs = {
        'mret': (_build_monthly_returns, ['ts_code', 'month', 'ret_fwd']),
        'mvind': (_build_mv_ind, ['ts_code', 'month', 'total_mv', 'sw_industry']),
        'flags': (lambda: pd.read_parquet(FLAGS), ['ts_code', 'month', 'is_one_word']),
    }
    for k, (fn, cols) in specs.items():
        cache = os.path.join(AUX_DIR, '%s.parquet' % k)
        if os.path.exists(cache):
            out[k] = pd.read_parquet(cache)
            continue
        try:
            d = fn()
        except Exception as e:
            log('  [辅助] %s 构建失败: %s' % (k, str(e)[:80]))
            d = None
        if d is not None:
            d.to_parquet(cache, index=False)
            out[k] = d
    return out


# ---------------------------------------------------------------- 列识别
def pick_col(df, cands, label):
    for c in cands:
        if c in df.columns:
            return c
    raise SystemExit('[错误] 未找到%s列，候选 %s；请用参数指定' % (label, cands))


def load_panel(path, args, aux):
    if path.lower().endswith('.csv'):
        df = pd.read_csv(path)
    else:
        df = pd.read_parquet(path)
    log('[载入] %s → %d 行 × %d 列' % (os.path.basename(path), len(df), df.shape[1]))

    code_col = args.code_col or pick_col(df, ['ts_code', 'code', 'symbol', 'stock_code', 'secid'], '代码')
    date_col = args.date_col or pick_col(df, ['month_end', 'date', 'trade_date', 'month', 'dt'], '日期')
    df = df.rename(columns={code_col: 'ts_code', date_col: 'month_end'})
    df['month'] = pd.to_datetime(df['month_end']).dt.to_period('M')

    # 收益
    ret_col = args.ret_col or next((c for c in ['ret_fwd', 'ret', 'fwd_ret', 'next_ret'] if c in df.columns), None)
    if ret_col and ret_col != 'ret_fwd':
        df = df.rename(columns={ret_col: 'ret_fwd'})
    if 'ret_fwd' not in df.columns:
        a = aux.get('mret')
        if a is None:
            raise SystemExit('[错误] 面板无收益列，且无法从行情构建')
        df = df.merge(a, on=['ts_code', 'month'], how='left')
        log('  [补] 下月收益 来自行情')
    df['ret_fwd'] = pd.to_numeric(df['ret_fwd'], errors='coerce').clip(-0.5, 1.0)

    # 市值 / 行业
    need_mv = not any(c in df.columns for c in ['V', 'total_mv', 'mv', 'mktcap'])
    need_ind = 'sw_industry' not in df.columns
    if need_mv or need_ind:
        a = aux.get('mvind')
        if a is not None:
            keep = ['ts_code', 'month'] + (['total_mv'] if need_mv else []) + (['sw_industry'] if need_ind else [])
            df = df.merge(a[keep], on=['ts_code', 'month'], how='left')
            log('  [补] %s' % ('+'.join(['市值'] if need_mv else [] + (['行业'] if need_ind else []))))
    mv_col = args.mv_col or next((c for c in ['V', 'total_mv', 'mv', 'mktcap'] if c in df.columns), None)
    if mv_col and mv_col != 'V':
        df = df.rename(columns={mv_col: 'V'})

    # 可成交标记
    if 'is_one_word' not in df.columns:
        a = aux.get('flags')
        if a is not None:
            df = df.merge(a, on=['ts_code', 'month'], how='left')
    if 'is_one_word' not in df.columns:
        df['is_one_word'] = np.nan
    if 'one_word_up' not in df.columns:
        df['one_word_up'] = False
    if 'one_word_down' not in df.columns:
        df['one_word_down'] = False
    df['suspended'] = df['is_one_word'].isna()
    df['one_word_up'] = df['one_word_up'].fillna(False).astype(bool)
    df['one_word_down'] = df['one_word_down'].fillna(False).astype(bool)

    if args.start:
        df = df[df['month'] >= pd.Period(args.start, 'M')]
    if args.end:
        df = df[df['month'] <= pd.Period(args.end, 'M')]
    return df


def detect_factors(df, args):
    if args.factors:
        cols = [c.strip() for c in args.factors.split(',') if c.strip()]
        miss = [c for c in cols if c not in df.columns]
        if miss:
            raise SystemExit('[错误] 因子列不存在: %s' % miss)
        return cols
    meta = {'ts_code', 'month_end', 'month', 'ret_fwd', 'mret', 'V', 'total_mv', 'sw_industry',
            'is_one_word', 'one_word_up', 'one_word_down', 'suspended', 'mv_src'}
    return [c for c in df.columns if c not in meta and pd.api.types.is_numeric_dtype(df[c])]


# ---------------------------------------------------------------- 体检函数
def hl_series(sub, col, weight='ew', exclude_untradable=False, q=Q):
    """返回 (月度多空收益 Series, 多头换手 list, 空头换手 list)。"""
    rows, to_l, to_s = [], [], []
    prev_l = prev_s = None
    for mt, g in sub.dropna(subset=[col, 'ret_fwd']).groupby('month'):
        g = g.copy()
        if exclude_untradable:
            g = g[~(g['one_word_up'] | g['one_word_down'] | g['suspended'])]
        if len(g) < 50:
            continue
        try:
            g['d'] = pd.qcut(g[col].rank(method='first'), q, labels=False) + 1
        except Exception:
            continue
        dd = {}
        for k in (1, q):
            m = g['d'] == k
            if not m.any():
                continue
            v = g.loc[m, 'ret_fwd'].values
            if weight == 'ew':
                dd[k] = v.mean()
            else:
                dd[k] = np.average(v, weights=np.clip(g.loc[m, 'V'].values, 1, None))
        if 1 not in dd or q not in dd:
            continue
        rows.append((mt, dd[q] - dd[1]))
        cur_l = set(g.loc[g['d'] == q, 'ts_code'])
        cur_s = set(g.loc[g['d'] == 1, 'ts_code'])
        if prev_l is not None:
            to_l.append(1 - len(prev_l & cur_l) / max(len(prev_l), 1))
            to_s.append(1 - len(prev_s & cur_s) / max(len(prev_s), 1))
        prev_l, prev_s = cur_l, cur_s
    s = pd.Series(dict(rows)).sort_index()
    return s, to_l, to_s


def ic_metrics(sub, col):
    ics, dec = [], {}
    for mt, g in sub.dropna(subset=[col, 'ret_fwd']).groupby('month'):
        if len(g) < 50:
            continue
        ic, _ = spearmanr(g[col], g['ret_fwd'])
        if not np.isnan(ic):
            ics.append(ic)
        gg = g.copy()
        try:
            gg['d'] = pd.qcut(gg[col].rank(method='first'), Q, labels=False) + 1
        except Exception:
            continue
        for k, v in gg.groupby('d')['ret_fwd'].mean().items():
            dec.setdefault(k, []).append(v)
    ics = pd.Series(ics).dropna()
    dm = pd.Series({k: np.mean(v) for k, v in dec.items()}).sort_index()
    mono = spearmanr(dm.index.values, dm.values)[0] if len(dm) >= 3 else np.nan
    rankic = ics.mean() if len(ics) else np.nan
    ic_t = rankic / (ics.std() / np.sqrt(len(ics))) if len(ics) > 1 and ics.std() else np.nan
    return {'RankIC': rankic, 'ICIR': rankic / ics.std() if len(ics) > 1 and ics.std() else np.nan,
            'IC_t': ic_t, 'IC>0占比': (ics > 0).mean() if len(ics) else np.nan,
            'N月': len(ics), '单调秩相关': mono}


def size_neutralize(sub, col):
    """逐月横截面 M ~ log(V) 取残差。"""
    ok = sub[col].notna() & sub['V'].notna() & (sub['V'] > 0)
    out = pd.Series(np.nan, index=sub.index, dtype=float)
    for mt, g in sub.loc[ok].groupby('month'):
        if len(g) < 20:
            continue
        x = np.log(g['V'].values)
        X = np.column_stack([np.ones(len(g)), x])
        y = g[col].values
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        out.loc[g.index] = y - X @ beta
    return out


def summarize_hl(s):
    if len(s) < 2 or s.std() == 0:
        return np.nan, np.nan
    return s.mean() * 100, s.mean() / (s.std() / np.sqrt(len(s)))


def verdict(r):
    """给单因子下结论。"""
    if not np.isfinite(r.get('IC_t', np.nan)):
        return '数据不足'
    if abs(r['IC_t']) < 1.5:
        return '❌ 否决：预测力不足(|IC_t|<1.5)'
    # 规模中性后符号反转
    if np.isfinite(r.get('中性化_t', np.nan)) and r['中性化_t'] * r['IC_t'] < 0:
        return '❌ 否决：规模中性后符号反转（疑似 size beta）'
    if np.isfinite(r.get('中性化_t', np.nan)) and abs(r['中性化_t']) < 1.0:
        return '⚠️ 存疑：规模中性后不显著'
    if np.isfinite(r.get('可成交衰减%', np.nan)) and r['可成交衰减%'] > 50:
        return '❌ 否决：可成交衰减 >50%（alpha 在不可交易股）'
    if np.isfinite(r.get('净_20bp', np.nan)) and r['净_20bp'] <= 0:
        return '⚠️ 存疑：扣成本后不赚钱'
    if abs(r['IC_t']) >= 3 and abs(r.get('中性化_t', 0)) >= 2 and r.get('可成交衰减%', 99) < 30:
        return '✅ 强力：三项全过'
    return '✅ 可用'


def main():
    ap = argparse.ArgumentParser(description='通用因子体检')
    ap.add_argument('--panel', required=True)
    ap.add_argument('--factors', default=None)
    ap.add_argument('--code-col', default=None)
    ap.add_argument('--date-col', default=None)
    ap.add_argument('--ret-col', default=None)
    ap.add_argument('--mv-col', default=None)
    ap.add_argument('--ind-col', default=None)
    ap.add_argument('--flip', action='store_true')
    ap.add_argument('--start', default=None)
    ap.add_argument('--end', default=None)
    ap.add_argument('--out', default='report_factor_health')
    args = ap.parse_args()

    log('===== 因子体检 =====')
    aux = ensure_aux()
    df = load_panel(args.panel, args, aux)
    factors = detect_factors(df, args)
    if not factors:
        raise SystemExit('[错误] 未识别到因子列，请用 --factors 指定')
    log('[因子] %d 个: %s' % (len(factors), ', '.join(factors)))
    log('[样本] %d 行 | %d 月 | %d 股 | %s..%s' % (
        len(df), df['month'].nunique(), df['ts_code'].nunique(),
        df['month'].min(), df['month'].max()))

    rows = []
    for f in factors:
        if args.flip:
            df['_f'] = -df[f]
            col = '_f'
        else:
            col = f
        m = ic_metrics(df, col)
        raw, tl, ts_ = hl_series(df, col, 'ew')
        vw, _, _ = hl_series(df, col, 'vw')
        neut = size_neutralize(df, col)
        df['_n'] = neut
        sn, _, _ = hl_series(df, '_n', 'ew')
        trad, tl2, ts2 = hl_series(df, col, 'ew', exclude_untradable=True)

        r0, t0 = summarize_hl(raw)
        rn, tn = summarize_hl(sn)
        rv, tv = summarize_hl(vw)
        rt, tt = summarize_hl(trad)
        to_avg = (np.mean(tl) + np.mean(ts_)) / 2 if tl else np.nan
        gross = raw.dropna()
        to_arr = np.array(tl) if tl else np.array([0.0])
        rec = {
            '因子': f, **{k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()},
            '多空_等权%': round(r0, 3), '多空t': round(t0, 2),
            '多空_市值加权%': round(rv, 3), '市值加权t': round(tv, 2),
            '中性化%': round(rn, 3), '中性化_t': round(tn, 2),
            '可成交%': round(rt, 3), '可成交t': round(tt, 2),
            '可成交衰减%': round(100 * (1 - rt / r0), 1) if r0 else np.nan,
            '月换手%': round(to_avg * 100, 1) if to_avg == to_avg else np.nan,
            '年化换手%': round(to_avg * 12 * 100, 0) if to_avg == to_avg else np.nan,
        }
        # 净收益：首月无换手（无前值），故收益从第 2 期起对齐换手序列
        gross_v = raw.dropna().values
        if len(tl) and len(tl) == len(gross_v) - 1:
            gross_v = gross_v[1:]
        cost_arr = (np.array(tl) + np.array(ts_)) if len(tl) == len(gross_v) else np.zeros(len(gross_v))
        for c in COSTS:
            net = gross_v - cost_arr * c * 2
            rec['净_%dbp' % (c * 10000)] = round(net.mean() * 100, 3)
        rec['结论'] = verdict(rec)
        rows.append(rec)
        log('  ✓ %s' % f)

    res = pd.DataFrame(rows)
    res.to_csv(args.out + '.csv', index=False, encoding='utf-8-sig')

    # ---------------- markdown 报告 ----------------
    L = []
    L.append('# 因子体检报告')
    L.append('')
    L.append('| 项目 | 值 |')
    L.append('|---|---|')
    L.append('| 面板 | `%s` |' % args.panel)
    L.append('| 样本 | %d 行 / %d 月 / %d 股 |' % (
        len(df), df['month'].nunique(), df['ts_code'].nunique()))
    L.append('| 区间 | %s ~ %s |' % (df['month'].min(), df['month'].max()))
    L.append('| 因子数 | %d |' % len(factors))
    L.append('| 生成时间 | %s |' % pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'))
    if args.flip:
        L.append('| 方向 | 已翻转（原值越小越好） |')
    L.append('')

    L.append('## 一、体检总览')
    L.append('')
    L.append('| 因子 | RankIC | IC_t | 单调性 | 规模中性后 t | 可成交衰减% | 净@20bp | 年化换手% | 结论 |')
    L.append('|---|---|---|---|---|---|---|---|---|')
    for r in rows:
        L.append('| %s | %.4f | %.2f | %.3f | %.2f | %s | %.3f | %s | %s |' % (
            r['因子'], r.get('RankIC', np.nan), r.get('IC_t', np.nan), r.get('单调秩相关', np.nan),
            r.get('中性化_t', np.nan),
            ('%.1f' % r['可成交衰减%']) if r['可成交衰减%'] == r['可成交衰减%'] else '-',
            r.get('净_20bp', np.nan),
            ('%.0f' % r['年化换手%']) if r['年化换手%'] == r['年化换手%'] else '-',
            r['结论']))
    L.append('')

    L.append('## 二、明细')
    L.append('')
    for r in rows:
        L.append('### %s' % r['因子'])
        L.append('')
        L.append('| 检验 | 指标 | 值 | 判定 |')
        L.append('|---|---|---|---|')
        L.append('| ① 预测力 | RankIC / ICIR / IC_t | %.4f / %.3f / %.2f | %s |' % (
            r.get('RankIC', np.nan), r.get('ICIR', np.nan), r.get('IC_t', np.nan),
            '通过' if abs(r.get('IC_t', 0)) >= 2 else '不足'))
        L.append('| ① 预测力 | 单调秩相关 | %.3f | %s |' % (
            r.get('单调秩相关', np.nan), '通过' if r.get('单调秩相关', 0) >= 0.8 else '弱'))
        L.append('| ② 规模中性 | 原始→中性化 H-L | %+.3f → %+.3f (t=%.2f) | %s |' % (
            r['多空_等权%'], r['中性化%'], r['中性化_t'],
            '符号反转(疑 size beta)' if r['中性化%'] * r['多空_等权%'] < 0 else
            ('保留' if abs(r['中性化_t']) >= 1.5 else '减弱')))
        L.append('| ② 规模中性 | 市值加权 H-L | %+.3f (t=%.2f) | %s |' % (
            r['多空_市值加权%'], r['市值加权t'],
            '稳健' if r['多空_市值加权%'] > 0 and abs(r['市值加权t']) >= 1.5 else '偏小盘依赖'))
        L.append('| ③ 换手成本 | 月/年化换手 | %.1f%% / %.0f%% | %s |' % (
            r['月换手%'], r['年化换手%'],
            '低' if r['年化换手%'] < 400 else ('中' if r['年化换手%'] < 800 else '偏高')))
        L.append('| ③ 换手成本 | 净收益 @10/20/30bp | %+.3f / %+.3f / %+.3f | %s |' % (
            r.get('净_10bp', np.nan), r.get('净_20bp', np.nan), r.get('净_30bp', np.nan),
            '通过' if r.get('净_20bp', 0) > 0 else '不赚钱'))
        L.append('| ④ 可成交性 | 剔除后 H-L (t) | %+.3f (t=%.2f) | %s |' % (
            r['可成交%'], r['可成交t'],
            '通过' if (r['可成交衰减%'] == r['可成交衰减%'] and r['可成交衰减%'] < 40) else '衰减大'))
        L.append('')

    L.append('## 三、体检口径说明')
    L.append('')
    L.append('- **① 预测力**：逐月横截面 Spearman RankIC；单调性=10 档平均收益与档位的秩相关')
    L.append('- **② 规模中性**：逐月横截面回归 `factor ~ log(总市值)` 取残差后重算；**符号反转=疑似 size beta**')
    L.append('- **③ 换手成本**：十分位多空月度换手；净收益 = 毛收益 − (多头换手+空头换手) × 2 × 单边成本')
    L.append('- **④ 可成交性**：剔除调仓日「停牌 / 一字涨跌停」后重算；衰减 >50% 视为不可实盘')
    L.append('- 组合口径：等权十分位多空（除标注市值加权处）；未考虑冲击成本与部分成交')
    L.append('')

    with open(args.out + '.md', 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(L))

    log('')
    log('===== 体检结论 =====')
    for r in rows:
        log('  %-16s IC_t=%5.2f  中性t=%5.2f  衰减=%s%%  净@20bp=%+.3f  → %s' % (
            r['因子'], r.get('IC_t', np.nan), r.get('中性化_t', np.nan),
            ('%.0f' % r['可成交衰减%']) if r['可成交衰减%'] == r['可成交衰减%'] else '-',
            r.get('净_20bp', np.nan), r['结论']))
    log('')
    log('已写 %s.md / %s.csv' % (args.out, args.out))


if __name__ == '__main__':
    main()
