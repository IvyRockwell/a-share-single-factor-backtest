# A 股单因子选股与回测研究系统

**A-Share Single-Factor Stock Selection & Backtesting System**

[![tests](https://github.com/IvyRockwell/a-share-single-factor-backtest/actions/workflows/tests.yml/badge.svg)](https://github.com/IvyRockwell/a-share-single-factor-backtest/actions/workflows/tests.yml)

一个入门级量化研究项目：用 CSMAR A 股数据检验价值（PE）、动量（Momentum）、质量（ROE）三类基础因子的选股能力，并把单因子研究结果转化为可回测的月度调仓组合。项目强调可复核与可复现——每个阶段都有固定的研究规则、可审计的中间产物、独立复算的校验和对应的测试。

**数据 → 因子 → 因子检验 → 选股 → 组合 → 回测 → 风险评价 → 研究报告**

## 核心研究问题

1. PE、Momentum 和 ROE 因子在 A 股市场中是否具有选股能力？
2. 基于这些因子的简单单因子策略，在考虑交易成本后能否取得较好的风险收益表现？
3. 研究结果是否会因股票选择范围、调仓频率或交易成本变化而明显改变？

## 主要结果

样本区间 2014-09-12 至 2025-12-31，136 个月末信号日，2,735 个交易日回测。完整分析见 [`reports/final_report.md`](reports/final_report.md)。

### 因子有效性

| 因子 | 累计 Q5-Q1 | 平均 Rank IC | IC 胜率 | 有效 IC 期数 |
| --- | ---: | ---: | ---: | ---: |
| PE | 76.70% | 0.0501 | 65.93% | 135 |
| Momentum | -74.23% | -0.0236 | 39.53% | 129 |
| ROE | -43.26% | 0.0065 | 48.89% | 135 |

![PE 因子检验：五分位、Q5-Q1 与 Rank IC](reports/figures/02_factor_pe.png)

### 策略回测（毛收益，Top 20% 等权，月度调仓）

| 序列 | 累计收益 | 年化收益 | 年化波动率 | 最大回撤 | Sharpe |
| --- | ---: | ---: | ---: | ---: | ---: |
| PE | 286.89% | 13.28% | 23.01% | -44.08% | 0.658 |
| Momentum | 25.76% | 2.13% | 25.50% | -62.32% | 0.211 |
| ROE | 117.99% | 7.44% | 24.13% | -53.71% | 0.419 |
| H00300（全收益基准） | 141.94% | 8.48% | 21.80% | -46.06% | 0.483 |

PE 的三项证据方向一致，但受下面「已知限制」中 PE 可用时点假设的约束；Momentum 的预设方向未得到因子检验支持，且对交易成本敏感；ROE 取得正收益但横截面检验证据较弱。加入 20 bp 滑点后，三者净累计收益分别为 255.64%、-11.77%、105.49%。

![三因子毛净值与 H00300](reports/figures/05_gross_nav_vs_benchmark.png)

三个毛收益策略都经历过较深回撤：PE -44.08%、Momentum -62.32%、ROE -53.71%，同期基准为 -46.06%。

![三因子策略与 H00300 的历史回撤](reports/figures/07_drawdowns.png)

## 研究流程

```text
CSMAR 原始数据
    ↓
数据清洗与标准化
    ↓
建立历史股票池（point-in-time，避免幸存者偏差）
    ↓
计算 PE / Momentum / ROE
    ↓
横截面排名与五分位分组
    ↓
因子有效性检验：Rank IC / 分组收益 / Q5-Q1
    ↓
构建单因子等权组合
    ↓
月度历史回测（下一交易日开盘执行）
    ↓
加入交易成本（佣金 / 印花税 / 滑点）
    ↓
风险收益评价
    ↓
稳健性测试（Top 10/20/30%、Monthly/Quarterly、0/10/20 bp）
    ↓
研究图表与最终研究报告
```

各阶段的研究规则记录在 `reports/pre_stage*_rule_confirmation.md`、`reports/stage5_research_plan.md` 和各阶段报告的开头。

## 数据来源与许可

本项目使用 **CSMAR（国泰安）A 股数据库**，原始导出文件由使用者自行从 CSMAR 获取，并放入项目根目录的 `项目初始数据/`。

**本仓库不包含任何 CSMAR 原始数据。** `项目初始数据/` 与 `data/processed/` 均不纳入版本控制；早期提交中曾包含两个阶段 3 的产物文件（`historical_universe.parquet`、`universe_exclusions.parquet`），当前树中已移除。CSMAR 数据受其最终用户许可协议约束，版权归深圳希施玛数据科技有限公司所有，未经授权不得复制、转让或再分发；使用前请确认你已获得合法授权。

用到的 CSMAR 表：

| 表 | 用途 |
| --- | --- |
| `TRD_Dalyr` | 个股日交易行情（收益、复权收盘价、交易状态） |
| `STK_MKT_ValuationMetrics` | 估值指标（`PE1TTM`） |
| `FS_Comins` / `FS_Combas` | 利润表与资产负债表（ROE） |
| `IAR_Forecdt` | 财务报告披露日期 |
| `TRD_Co` / `STK_LISTEDCOINFOANL` / `STK_ITEMCHANGE` | 股票基本信息、上市日期、上市状态变更 |
| `TRD_Cale` | 交易日历 |
| `IDX_Idxtrd` | 沪深 300 全收益指数（H00300） |

字段映射与口径见 `data/processed/field_mapping.csv`（阶段 2 生成）和 [`reports/data_audit_stage1.md`](reports/data_audit_stage1.md)。

## 环境要求

- Python 3.10 或更高版本（开发与验证环境为 3.13）
- 依赖见 [`requirements.txt`](requirements.txt)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS / Linux: source .venv/bin/activate
pip install -r requirements.txt
```

图表中文标签使用 `Microsoft YaHei` / `SimHei`。在 macOS 或 Linux 上重跑阶段 11 时，需要在 [`src/visualization.py`](src/visualization.py) 的 `font.sans-serif` 中加入系统已有的中文字体，否则中文会显示为方框。

## 快速开始

1. 把 CSMAR 导出的 CSV（保留 `[DES][csv].txt` 字段说明文件）按原目录结构放入 `项目初始数据/`。
2. 依次运行各阶段。`run.py` 不带参数时只打印帮助，不会自动执行任何阶段：

```bash
# 阶段 1：原始数据审计（Notebook）
jupyter execute --inplace --timeout=-1 notebooks/01_data_exploration.ipynb

# 阶段 2～11：数据清洗 → 股票池 → 因子 → 检验 → 组合 → 回测 → 成本 → 绩效 → 稳健性 → 图表
python run.py stage2
python run.py stage3
python run.py stage4
python run.py stage5
python run.py stage6
python run.py stage7
python run.py stage8
python run.py stage9
python run.py stage10
python run.py stage11
```

3. 运行测试：

```bash
python -m pytest -q
```

测试使用构造数据，不依赖 `项目初始数据/` 或 `data/processed/`，可在干净环境中直接运行。

阶段 2 读取全部原始 CSV，耗时较长；阶段 10 会跑 54 个参数组合，是耗时最长的一步。最终报告 [`reports/final_report.md`](reports/final_report.md) 读取阶段 4～11 已固化的产物，不重新选择参数或改变研究规则。

## 项目结构

```text
.
├── configs/
│   └── config.yaml              # 全部研究参数与研究假设（含来源）
├── data/                        # 本地生成，不纳入版本控制
│   └── processed/               # 阶段 2～10 的中间产物
├── notebooks/
│   └── 01_data_exploration.ipynb
├── reports/
│   ├── figures/                 # 17 张研究图表
│   ├── final_report.md          # 最终研究报告
│   ├── final_audit.md           # 终审报告：问题、证据、修复与复核
│   ├── stage11_figure_index.md  # 图表索引与口径
│   └── *.md                     # 各阶段报告与研究规则确认
├── src/
│   ├── data_loader.py           # 阶段 2：清洗与标准化
│   ├── universe.py              # 阶段 3：历史股票池
│   ├── factors.py               # 阶段 4：PE / Momentum / ROE
│   ├── factor_test.py           # 阶段 5：分组、未来收益、Rank IC
│   ├── portfolio.py             # 阶段 6：目标组合与调仓动作
│   ├── backtest.py              # 阶段 7：毛收益回测
│   ├── costs.py                 # 阶段 8：佣金、印花税、滑点
│   ├── performance.py           # 阶段 9：收益与风险指标
│   ├── robustness.py            # 阶段 10：稳健性网格
│   └── visualization.py         # 阶段 11：图表
├── .github/workflows/tests.yml   # CI：push / PR 自动运行全部测试
├── tests/                       # 11 个测试模块 + conftest.py，51 个用例
├── run.py                       # 阶段入口
├── requirements.txt
├── pyproject.toml               # pytest 与 ruff 配置
├── LICENSE
└── README.md
```

## 产物说明

| 位置 | 内容 | 是否入库 |
| --- | --- | --- |
| `项目初始数据/` | CSMAR 原始导出 | 否 |
| `data/processed/` | 标准化数据、股票池、因子、回测与绩效中间产物 | 否 |
| `reports/*.md` | 各阶段报告、终审报告、最终研究报告 | 是 |
| `reports/figures/*.png` | 17 张研究图表 | 是 |

`data/processed/` 中的文件名即阶段编号（如 `backtest_stage8_daily_returns.parquet`），可从文件名直接追溯到生成它的阶段。

## 稳健性测试

固定检验 Top 10% / 20% / 30%、Monthly / Quarterly 与 0 / 10 / 20 bp 滑点，三个因子合计 54 个策略成本情景。PE 与 ROE 在已测试网格内没有出现累计净收益正负反转（PE 为 189.41%～278.48%，ROE 为 62.43%～137.11%）；Momentum 在 Top 10% / Monthly / 10 bp，以及 Top 10% 或 Top 20% / Monthly / 20 bp 三个情景下转为负收益，其余情景仍为正。测试只描述敏感性，不据此重新选择参数。

![PE 稳健性测试](reports/figures/11_robustness_pe.png)

全部 17 张图表见 [`reports/figures/`](reports/figures)，每张图的数据来源与口径见 [`reports/stage11_figure_index.md`](reports/stage11_figure_index.md)。

## 已知限制

- **PE 前视风险**：`PE1TTM` 按交易日可用处理，原始数据无法复核其 TTM 财务分母的披露时点、报表版本与历史股本生效时间，因此 PE 结果不能表述为已严格消除前视偏差。
- **ROE**：只使用信号日前严格已披露的年度合并报表，权益必须为正。
- **Momentum**：过去 6 个自然月收益，不跳过最近一个月。
- **执行与成交**：策略在下一交易日开盘执行，基准使用完整交易日收益，盘中起点不完全一致。正常交易状态和有效开盘价不等于必然成交；涨跌停、盘口深度、最低佣金、整手约束和资金容量均未纳入。
- **交易成本**：佣金 3 bp 与滑点情景是研究假设，不代表任何券商的实际费率；0 bp 只表示无滑点，仍包含佣金和卖出印花税。
- 因子检验、相关性和历史回测都不能证明因果关系，样本内收益不能外推为未来收益。

## 免责声明

本项目仅用于研究与教学，不构成任何投资建议。历史回测结果不代表未来收益。

## 许可证

源代码与文档采用 [MIT License](LICENSE)。CSMAR 数据不在本许可证覆盖范围内，详见上文「数据来源与许可」。
