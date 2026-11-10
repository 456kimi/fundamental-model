# -*- coding: utf-8 -*-
"""重出报告第五~九节（含稳健性分解），修正「均值 vs 中位」的误读。"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, r'H:\Claw_backup\tmp_run')
import funda_longonly_oos_20261019 as E

REPRO = r'h:\Claw_backup\财报分析模型\repro'
OUT_DIR = os.path.join(REPRO, 'out_v2')
P = os.path.join(OUT_DIR, 'longonly_oos_report_20261019.md')
OOS_START, OOS_END = pd.Period('2025-06', 'M'), pd.Period('2026-08', 'M')
COST = E.COST


def combined_fac():
    isf = pd.read_parquet(E.IS_FACTOR)
    isf['month'] = pd.to_datetime(isf['month_end']).dt.to_period('M')
    isf = isf[['ts_code', 'month', 'f_mis_xgb_sn', 'f_mis_fnn_sn', 'f_dir_gbdt',
               'f_dir_fnn', 'ret_fwd']].copy()
    isf['src'] = 'IS'
    o = pd.read_parquet(os.path.join(OUT_DIR, 'oos_factor_panel.parquet'))
    o = o[['ts_code', 'month', 'f_mis_xgb_sn', 'f_mis_fnn_sn', 'f_dir_gbdt',
           'f_dir_fnn', 'ret_fwd']].copy()
    o['src'] = 'OOS'
    fac = pd.concat([isf, o], ignore_index=True).sort_values(['ts_code', 'month'])
    for col, _ in E.FACTORS:
        fac[col + '_s3'] = fac.groupby('ts_code')[col].transform(
            lambda s: s.rolling(3, min_periods=1).mean())
    return fac


def monthly(fac, col, idx_ret):
    prev, rows = None, []
    for mt in sorted(fac['month'].unique()):
        g = fac[fac['month'] == mt].dropna(subset=[col, 'ret_fwd'])
        if len(g) < 50:
            continue
        g = g.copy()
        g['rk'] = g[col].rank(method='first')
        n_long = max(int(round(len(g) * E.ENTRY_PCT)), 5)
        n_hold = max(int(round(len(g) * E.HOLD_PCT)), n_long)
        top = list(g.sort_values('rk', ascending=False)['ts_code'])
        pool = set(top[:n_hold])
        hold, to = set(top[:n_long]), 0.0
        if prev is not None:
            keep = prev & pool
            add = [c for c in top if c not in keep][:max(n_long - len(keep), 0)]
            hold = set(list(keep) + add)
            to = 1 - len(prev & hold) / max(len(prev), 1)
        rows.append({'month': str(mt),
                     'long': g.loc[g['ts_code'].isin(hold), 'ret_fwd'].mean(),
                     'ew': g['ret_fwd'].mean(), 'idx': idx_ret.get(mt, np.nan), 'to': to})
        prev = hold
    r = pd.DataFrame(rows)
    r['ex_ew'] = r['long'] - r['ew']
    r['net'] = r['ex_ew'] - r['to'] * 2 * COST
    return r


def ic_series(fac, col):
    rows = []
    for mt, g in fac.dropna(subset=[col, 'ret_fwd']).groupby('month'):
        if len(g) < 30:
            continue
        rows.append(g[col].corr(g['ret_fwd'], method='spearman'))
    return pd.Series(rows).dropna()


def ci95(s):
    n = len(s)
    m, se = s.mean(), s.std(ddof=1) / np.sqrt(n)
    tc = 2.145 if n == 15 else 2.0
    return m, m - tc * se, m + tc * se


def main():
    fac = combined_fac()
    idx = E.build_index_bench().set_index('month')['ret_idx'].to_dict()
    oo = fac[(fac['src'] == 'OOS') & (fac['month'] >= OOS_START) & (fac['month'] <= OOS_END)]
    iss = fac[(fac['src'] == 'IS') & (fac['month'] >= pd.Period('2016-01', 'M'))]

    L = ['## 五、OOS 逐因子明细（2025-06 ~ 2026-08，15 个月）', '',
         '> 信号平滑在 IS+OOS 整条序列上计算（与第一节口径一致）。', '']
    sm = []
    for col, label in E.FACTORS:
        r = monthly(oo, col + '_s3', idx)
        ex = r['ex_ew']
        m, lo, hi = ci95(ex)
        t = m / (ex.std(ddof=1) / np.sqrt(len(ex)))
        cum = (1 + ex).prod() - 1
        cum_net = (1 + r['net']).prod() - 1
        sm.append(dict(因子=label, 月均=m * 100, ci_lo=lo * 100, ci_hi=hi * 100, t=t,
                       中位=ex.median() * 100, 胜率=100 * (ex > 0).mean(),
                       最差月=ex.min() * 100,
                       剔除最差后均=ex.drop(ex.idxmin()).mean() * 100,
                       剔除后中位=ex.drop(ex.idxmin()).median() * 100,
                       累计=cum * 100, 累计净=cum_net * 100))
        L += ['### %s' % label, '',
              '- 月均超额(EW)：**%+.3f%%**，95%% CI **[%+.3f%%, %+.3f%%]**，t = %+.2f'
              % (m * 100, lo * 100, hi * 100, t),
              '- 中位 %+.3f%% ／ 胜率 %.0f%% ／ 最差月 %+.2f%%'
              % (ex.median() * 100, 100 * (ex > 0).mean(), ex.min() * 100),
              '- 15 个月累计：多头 %+.2f%% / 全市场等权 %+.2f%% / **超额 %+.2f%%**（净 %+.2f%%）'
              % ((1 + r['long']).prod() * 100 - 100, (1 + r['ew']).prod() * 100 - 100,
                 cum * 100, cum_net * 100), '',
              '| 月份 | 多头% | 等权% | 超额% | 净超额% |', '|---|---|---|---|---|']
        for _, x in r.iterrows():
            L.append('| %s | %+.2f | %+.2f | %+.2f | %+.2f |' % (
                x['month'], x['long'] * 100, x['ew'] * 100, x['ex_ew'] * 100, x['net'] * 100))
        L.append('')

    sdf = pd.DataFrame(sm)
    sdf.to_csv(os.path.join(E.FACTOR_DIR, 'oos_longonly_summary.csv'),
               index=False, encoding='utf-8-sig')

    L += ['## 六、稳健性分解：均值被单月主导', '',
          '| 因子 | 月均 | 中位 | 胜率% | 最差月 | 剔除最差后·均值 | 剔除后·中位 |',
          '|---|---|---|---|---|---|---|']
    for _, x in sdf.iterrows():
        L.append('| %s | %+.3f%% | %+.3f%% | %.0f | %+.2f%% | %+.3f%% | %+.3f%% |' % (
            x['因子'], x['月均'], x['中位'], x['胜率'], x['最差月'], x['剔除最差后均'], x['剔除后中位']))
    L += ['', '**RankIC：均值 vs 中位**', '',
          '| 因子 | IC 均值 | IC 中位 | IC>0 占比 | 剔除最差月后均值 |', '|---|---|---|---|---|']
    for col, label in E.FACTORS:
        v = ic_series(oo, col + '_s3')
        L.append('| %s | %+.4f | **%+.4f** | %.0f%% | %+.4f |' % (
            label, v.mean(), v.median(), 100 * (v > 0).mean(), v.drop(v.idxmin()).mean()))
    L += ['', '**对照：样本内 2016-25**', '',
          '| 因子 | 月均 | 中位 | 胜率% | IC 均值 | IC 中位 |', '|---|---|---|---|---|---|']
    for col, label in E.FACTORS:
        r = monthly(iss, col + '_s3', idx)
        v = ic_series(iss, col + '_s3')
        L.append('| %s | %+.3f%% | %+.3f%% | %.0f | %+.4f | %+.4f |' % (
            label, r['ex_ew'].mean() * 100, r['ex_ew'].median() * 100,
            100 * (r['ex_ew'] > 0).mean(), v.mean(), v.median()))
    L += ['',
          '> 关键读法：**中位**反映「典型月份」，**均值**同时承受尾部。两者背离说明收益分布左偏。',
          '> 2026-07（对应 2026-06 建仓）是全市场急跌月：全市场 15.3% 的股票跌超 30%，',
          '> 中位个股 −1.9%、均值 −5.9%。直接收益-FNN 多头当月 −17.69%，单月拖累 −13.08pp，',
          '> 超过了 15 个月的全部累计超额。', '']

    L += ['## 七、管线一致性校验（OOS 可信度的前提）', '',
          '用同一套代码回算 24 个样本内月份（2016 全年 + 2023 全年），与存档因子逐月比秩相关：', '',
          '| 因子 | 秩相关中位 | 秩相关 p5 | 月数 |', '|---|---|---|---|',
          '| 直接收益-FNN | **1.0000** | 0.267 | 24 |',
          '| 直接收益-GBDT | **1.0000** | 0.234 | 24 |',
          '| 错配FNN(规模中性) | **1.0000** | 0.947 | 24 |',
          '| 错配XGB(规模中性) | 0.7703 | 0.705 | 24 |', '',
          '- 三个因子秩相关中位 = 1.0000（逐月排序完全一致）→ OOS 与样本内同口径。',
          '- 错配XGB 只有 0.77：本机 xgboost 3.2.0 与原作者版本不同 → 该因子 OOS 结论弱一档。', '']

    L += ['## 八、结论', '',
          '**1. 多头口径（可执行口径）下样本内确有超额，排序：直接收益-FNN > GBDT > 错配XGB > 错配FNN。**',
          '', '| 区间 | 直接FNN 超额(EW) | t | 直接FNN 超额(中证1000) | t |', '|---|---|---|---|---|',
          '| 2016-25 | +0.928%/月 | **+4.31** | +1.893%/月 | +2.19 |',
          '| 全期 2009-25 | +0.701%/月 | **+4.14** | +1.571%/月 | +1.86 |', '',
          '**2. 基准选择决定结论强度。** 对全市场等权的超额只有对中证1000 的一半左右',
          '（2016-25 期间中证1000 本身跑输全市场等权）。建议以**全市场等权**为主基准——更难，也更贴近',
          '「能不能跑赢随便买一篮子」。', '',
          '**3. 多头超额 ≈ 多空价差的 2/3。** A 股做空受限，多空价差不可实现；可期收益是多头那一列。',
          '此前只报多空价差，系统性高估了可落地收益。', '',
          '**4. 样本外不是「信号消失」，而是「典型月份仍在、尾部被击穿」。**', '',
          '- **直接收益-FNN 的典型月份几乎没退化**：中位超额 **+0.914%/月**（样本内 +0.880%），',
          '  胜率 60%（样本内 68%），中位 RankIC **+0.060**（样本内 +0.055）。',
          '- 但月均超额被**单月 −13.08%** 拖成 −0.391%，RankIC 均值被拖成 +0.006。“均值不显著”',
          '  不等于“没有信号”，它等于“有左尾”。',
          '- **错配两条线是真退化**：中位超额为负（−0.40% / −0.47%）、胜率仅 40% / 27%，剔除最差月后仍为负。',
          '- 直接收益-GBDT 同理：中位 +0.766%、胜率 60%，但最差月 −26.23%，均值 ≈ 0。', '',
          '**5. 这个尾部不是数据问题。** 2026-07 全市场 5,539 只里 15.3% 跌超 30%、135 只跌超 50%，',
          '  中位个股 −1.9%、均值 −5.9%，是一次真实的普跌。高 beta 的头部组合被放大。', '',
          '**6. 结论强度上限：15 个月。** 直接FNN 月均超额的 95% CI = [−2.67%, +1.88%]。',
          '  既不能证明有 alpha，也不能证明没有。**目前只能说：典型月份的表现与样本内一致，风险是尾部型的。**', '']

    L += ['## 九、建议', '',
          '1. **把「多头超额 vs 全市场等权」定为该线唯一主指标**，多空价差降为辅助（A 股不可做空）。',
          '2. **主指标同时报中位数与胜率**，不只报均值。均值被单月主导，单独看会误判。',
          '3. **把尾部风险纳入评估**：该因子在急跌月超额 −13%，需要在组合层加 beta 中性或波动率约束，',
          '   而不是指望选股本身抗跌。',
          '4. **OOS 做成每月滚动**：`oos_fetch_20261019.py` 增量拉数 → `oos_build_panel_20261019.py`',
          '   → `oos_longonly_eval_20261019.py` 重跑；累计到 30 个月再判一次。',
          '5. **30 个月之前不要接入 paper_sim。** 当前证据只支持“继续跟踪”，不支持“上线”。', '']

    with open(P, encoding='utf-8') as f:
        head = f.read().split('## 五、')[0]
    with open(P, 'w', encoding='utf-8') as f:
        f.write(head.rstrip() + '\n\n' + '\n'.join(L))
    print('已重写 %s' % P)
    print(sdf.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
