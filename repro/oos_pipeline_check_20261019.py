# -*- coding: utf-8 -*-
"""OOS 建模管线的一致性校验。

问题：OOS 因子（2025-06 起）由本会话代码重建，样本内因子来自
      factors/funda_ml_factor_panel.parquet（原作者产出）。
      若两套口径不同，OOS 归零可能只是代码差异造成的假象。
做法：用同一套 OOS 代码回算若干样本内月份，与存档因子逐月比秩相关。
      秩相关应接近 1（模型随机种子/库版本差异允许轻微偏离）。
"""
import io
import os
import sys
import numpy as np
import pandas as pd

# 注意：不要在这里包 sys.stdout——被导入的模块会在 import 时自带一层包装，
# 双重包装会导致 GC 时关闭底层 buffer（HANDOFF 里记过的坑）。
sys.path.insert(0, r'H:\Claw_backup\tmp_run')
import funda_longonly_oos_20261019 as E

REPRO = r'h:\Claw_backup\财报分析模型\repro'
IS_FACTOR = os.path.join(REPRO, 'factors', 'funda_ml_factor_panel.parquet')


def main():
    print('=' * 78)
    print('### 用 OOS 管线回算样本内月份，与存档因子比对')
    print('=' * 78)
    df = E.load_combined()
    all_months = sorted(df['month'].unique())

    checks = [pd.Period('2016-%02d' % m, 'M') for m in range(1, 13)] + \
             [pd.Period('2023-%02d' % m, 'M') for m in range(1, 13)]
    targets = [m for m in checks if m in all_months]
    print('  回算月份: %s .. %s (%d 个)' % (targets[0], targets[-1], len(targets)))

    regen = E.gen_oos(df, targets)
    isf = pd.read_parquet(IS_FACTOR)
    isf['month'] = pd.to_datetime(isf['month_end']).dt.to_period('M')
    isf = isf[isf['month'].isin(targets)]

    pairs = [('f_dir_fnn', '直接收益-FNN'), ('f_dir_gbdt', '直接收益-GBDT'),
             ('f_mis_xgb_sn', '错配XGB(规模中性)'), ('f_mis_fnn_sn', '错配FNN(规模中性)')]
    print('\n  %-22s %10s %10s %10s %10s' % ('因子', '秩相关中位', '秩相关p5', '值相关中位', '月数'))
    for col, label in pairs:
        rs, vs = [], []
        for mt in targets:
            a = regen[regen['month'] == mt][['ts_code', col]].dropna()
            b = isf[isf['month'] == mt][['ts_code', col]].dropna()
            j = a.merge(b, on='ts_code', suffixes=('_new', '_old'))
            if len(j) < 100:
                continue
            rs.append(j[col + '_new'].corr(j[col + '_old'], method='spearman'))
            vs.append(j[col + '_new'].corr(j[col + '_old']))
        if rs:
            print('  %-22s %10.4f %10.4f %10.4f %10d' % (
                label, np.median(rs), np.percentile(rs, 5), np.median(vs), len(rs)))

    # 关键：用回算的样本内因子跑一遍评估，看结论是否与存档一致
    print('\n  对照：用「回算因子」在 2016-25 上重算多空，验证结论方向')
    isf_all = pd.read_parquet(IS_FACTOR)
    isf_all['month'] = pd.to_datetime(isf_all['month_end']).dt.to_period('M')
    isf_all = isf_all[isf_all['month'] >= pd.Period('2016-01', 'M')]
    print('  %-22s %14s %14s' % ('因子', '存档 多空(t)', '回算 多空(t)'))
    for col, label in pairs:
        hm, ht = E.hl_metrics(isf_all, col)
        # 回算集只覆盖 2016/2023 两年，仅看方向
        hm2, ht2 = E.hl_metrics(regen, col)
        print('  %-22s %+7.3f(%5.2f) %+7.3f(%5.2f)  [回算仅 24 个月]'
              % (label, hm * 100, ht, hm2 * 100, ht2))


if __name__ == '__main__':
    main()
