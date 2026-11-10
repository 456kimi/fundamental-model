# -*- coding: utf-8 -*-
"""因子质检（RankIC / ICIR / 十分位单调性）+ 生成口径登记 README。

输入 : factors/funda_ml_factor_panel.parquet
输出 : factors/factor_quality.csv
       factors/README.md
"""
import io, sys, os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

FACTOR_DIR = r'h:\Claw_backup\财报分析模型\repro\factors'
PQ = os.path.join(FACTOR_DIR, 'funda_ml_factor_panel.parquet')

FACTORS = {
    'f_mis_xgb': '估值错配-XGBoost',
    'f_mis_xgb_sn': '估值错配-XGBoost(规模中性)',
    'f_mis_fnn': '估值错配-FNN',
    'f_mis_fnn_sn': '估值错配-FNN(规模中性)',
    'f_dir_gbdt': '直接收益-GBDT',
    'f_dir_fnn': '直接收益-FNN',
}

df = pd.read_parquet(PQ)
df['month'] = pd.to_datetime(df['month_end']).dt.to_period('M')
df['year'] = df['month'].astype(str).str[:4].astype(int)
print('因子面板 %d 行 | %d 月 | %d 股 | %s..%s' % (
    len(df), df['month'].nunique(), df['ts_code'].nunique(),
    df['month'].min(), df['month'].max()))


def metrics(sub, col):
    ics, mono = [], []
    for mt, g in sub.dropna(subset=[col, 'ret_fwd']).groupby('month'):
        if len(g) < 50:
            continue
        ic, _ = spearmanr(g[col], g['ret_fwd'])
        if not np.isnan(ic):
            ics.append(ic)
        g = g.copy()
        try:
            g['d'] = pd.qcut(g[col].rank(method='first'), 10, labels=False) + 1
            dd = g.groupby('d')['ret_fwd'].mean()
            if 10 in dd.index and 1 in dd.index:
                mono.append(dd[10] - dd[1])
        except Exception:
            pass
    ics = pd.Series(ics).dropna()
    mono = pd.Series(mono).dropna()
    if len(ics) < 2:
        return None
    icir = ics.mean() / ics.std() if ics.std() else np.nan
    pos = (ics > 0).mean()
    # 单调性：10 档收益的秩相关
    dec = {}
    for mt, g in sub.dropna(subset=[col, 'ret_fwd']).groupby('month'):
        if len(g) < 50:
            continue
        g = g.copy()
        try:
            g['d'] = pd.qcut(g[col].rank(method='first'), 10, labels=False) + 1
        except Exception:
            continue
        for k, v in g.groupby('d')['ret_fwd'].mean().items():
            dec.setdefault(k, []).append(v)
    dm = pd.Series({k: np.mean(v) for k, v in dec.items()}).sort_index()
    mono_rank = spearmanr(dm.index.values, dm.values)[0] if len(dm) >= 3 else np.nan
    return {
        'RankIC': round(ics.mean(), 4), 'ICIR': round(icir, 3),
        'IC_t': round(ics.mean() / (ics.std() / np.sqrt(len(ics))), 2),
        'IC>0占比': round(pos, 3), 'N月': len(ics),
        '多空月均%': round(mono.mean() * 100, 3) if len(mono) else np.nan,
        '多空t': round(mono.mean() / (mono.std() / np.sqrt(len(mono))), 2) if len(mono) and mono.std() else np.nan,
        '单调秩相关': round(mono_rank, 3),
    }


rows = []
for c, label in FACTORS.items():
    if c not in df.columns:
        continue
    for tag, sub in [('全期', df),
                     ('2009-15', df[df.year <= 2015]),
                     ('2016-25', df[df.year >= 2016])]:
        m = metrics(sub, c)
        if m:
            rows.append(dict(因子=label, 区间=tag, **m))

q = pd.DataFrame(rows)
q.to_csv(os.path.join(FACTOR_DIR, 'factor_quality.csv'), index=False, encoding='utf-8-sig')
print()
print(q.to_string(index=False))

# ---------------- README ----------------
best = q[(q['区间'] == '2016-25')].sort_values('IC_t', ascending=False)
readme = f"""# 基本面 ML 因子面板（funda_ml_factor_panel）

> 来源：复现《基于机器学习的企业基本面估值错配异象研究》（系统工程理论与实践 2025, 45(9)）
> 生成脚本：`repro/dual_filter_and_factor.py`　质检：`repro/factor_quality.py`
> 生成日期：{pd.Timestamp.now():%Y-%m-%d}

## 一、文件

| 文件 | 说明 |
|---|---|
| `funda_ml_factor_panel.parquet` | 因子面板，{len(df)} 行 × 8 因子列 |
| `factor_quality.csv` | RankIC / ICIR / 单调性质检 |

## 二、因子字段

| 字段 | 含义 | 方向 |
|---|---|---|
| `f_mis_xgb` | 估值错配指标（XGBoost 预测内在估值 vs 真实市值） | 值越大 → 越低估 → 预期收益越高 |
| `f_mis_xgb_sn` | 同上，**规模中性化** | 同上 |
| `f_mis_fnn` | 估值错配指标（FNN） | 同上 |
| `f_mis_fnn_sn` | 同上，规模中性化 | 同上 |
| `f_dir_gbdt` | **直接预测下月收益**（GBDT） | 值越大 → 预期收益越高 |
| `f_dir_fnn` | 直接预测下月收益（FNN） | 同上 |
| `ret_fwd` | 下月实际收益（评估用） | — |
| `V` | 月末总市值 | — |

## 三、构建口径（预注册，改动须登记）

- **样本**：沪深两市 A 股，剔除科创板（688）、北交所；剔除净资产为负
- **区间**：2009-01 ~ 2025-05，191 个预测月
- **特征**：37 个基本面变量（35 财务 + 供应链代理 2），来源 Tushare 2008 年起
- **训练**：滑动窗口 5 个月，仅用 t-5..t-1 历史，**逐月滚动**
- **无前视**：`ann_visible <= month_end` 硬校验通过（0 违规）；模型每月重训
- **目标**：估值错配路线目标 = log(总市值)；直接路线目标 = 下月收益
- **中性化**：逐月横截面回归取残差（规模中性：log V；规模+行业：log V + 申万行业哑变量）

## 四、质检结果（2016-25 段，按 IC_t 排序）

| 因子 | RankIC | ICIR | IC_t | 多空月均% | 多空t |
|---|---|---|---|---|---|
"""
for _, r in best.iterrows():
    readme += '| %s | %.4f | %.3f | %.2f | %s | %s |\n' % (
        r['因子'], r['RankIC'], r['ICIR'], r['IC_t'], r['多空月均%'], r['多空t'])

readme += """
## 五、稳健性结论（双重过滤）

**2016-25 段（最可交易）**：

| 因子 | 等权+规模中性 | 等权+规模&行业中性 | 市值加权+规模&行业中性 |
|---|---|---|---|
| 估值错配-XGBoost | +0.91 (3.21) | +1.01 (4.80) | **+0.88 (2.59)** |
| 直接收益-GBDT | +1.30 (3.23) | +1.44 (4.77) | +0.89 (2.00) |
| 估值错配-FNN | +1.02 (3.09) | +0.93 (3.47) | +0.24 (1.06) |

**结论**：
1. **剥离规模后信号仍在**——非线性因子不是规模溢价的换皮
2. **XGBoost 是唯一通过最严过滤（市值加权+规模&行业中性）的因子**（+0.88%/月，t=2.59）
3. **2009-15 段剥离规模后基本失效**（t<2）——异象主要存在于近十年
4. **直接收益路线 ≥ 估值错配路线**，且实现更简单

## 六、风险与限制（必读）

- 等权组合收益约 **60% 来自小盘股暴露**；市值加权后收益缩水至约一半
- 未做交易成本、涨跌停不可成交、停牌过滤 → **实盘收益应低于样本内**
- 特征含 2 个行情代理（NEWS/MEDIA），非纯基本面 → 与论文口径存在差异
- 未做 FF3/FF5/CH-4 风险调整（本地无因子收益序列）
- 因子为**月度调仓**，换手率未评估
- 群聊/舆情类外部数据未纳入本面板
"""

with open(os.path.join(FACTOR_DIR, 'README.md'), 'w', encoding='utf-8') as f:
    f.write(readme)
print()
print('已写 factors/README.md 与 factors/factor_quality.csv')
