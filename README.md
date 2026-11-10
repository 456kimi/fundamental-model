# 财报分析模型

复现《基于机器学习的企业基本面估值错配异象研究》（《系统工程理论与实践》2025, 45(9): 2912–2933）。

核心问题：用基本面变量经非线性 ML 估计企业的**内在估值**，与真实市值比对，错配部分是否构成可交易的因子。

## 流水线

```
财报三表 + 行情  →  37 个基本面变量面板  →  ML 估值预测  →  错配因子  →  多空 / 多头超额回测
     fetch_*            build_*_panel           repro_funda*       dual_filter_and_factor    oos_*
```

| 阶段 | 脚本 |
|------|------|
| 数据抓取 | `fetch_fin_ext.py`（财报三表）、`fetch_mkt_ext.py`（日线/市值/复权）、`oos_fetch_20261019.py`（样本外窗口） |
| 面板构建 | `build_monthly_panel.py`、`build_panel_ext.py`、`build_debt_detail_20261023.py` |
| 因子构造 | `repro_funda.py`、`repro_funda_v2.py`、`dual_filter_and_factor.py`、`factor_quality.py` |
| 样本外检验 | `oos_*`（评估 / 稳健性 / 规模中性 / 尾部诊断 / 图表） |
| 稳健性 | `size_neutral_check.py`、`size_neutral_subperiod.py`、`turnover_optimize.py`、`robust_and_direct.py`、`tradability.py` |
| 工具 | `tools/factor_health_check.py`（因子健康度自检） |

## 关于本仓库

**只含代码与文档，不含数据。** 财务面板、行情数据、中间产物均在本地，注册在 `.gitignore`。

克隆后需要自备 Tushare 凭证：

1. 环境变量方式：`set TUSHARE_TOKEN=你的token`
2. 或新建 `repro/_secrets_local.py`，内容 `TOKEN = "你的token"`

`_secrets.py` 会按「环境变量 → 本地私有文件」的顺序读取，仓库里不含任何真实密钥。

## 文档

- `WORKLOG.md` — 迭代记录与口径变更
- `研究备忘录_基本面ML估值错配复现_20261019.md` — 完整复现报告（因子构造、分组收益、结论）
- `refs/papers/_extracted.txt` — 原论文抽取文本
