# A 股单因子选股与回测研究系统

## 项目名称

**A-Share Single-Factor Stock Selection & Backtesting System**

## 项目目标

使用 CSMAR A 股数据，研究价值、动量、质量三类基础因子的选股有效性，并将单因子研究结果转化为简单的月度调仓组合，完成历史回测、交易成本分析、稳健性测试和研究报告。

本项目是入门级量化研究项目，重点是完整、可复现地走通：

**数据 → 因子 → 因子检验 → 选股 → 组合 → 回测 → 风险评价 → 研究报告**

## 核心研究问题

1. PE、Momentum 和 ROE 因子在 A 股市场中是否具有选股能力？
2. 基于这些因子的简单单因子策略，在考虑交易成本后能否取得较好的风险收益表现？
3. 研究结果是否会因股票选择范围、调仓频率或交易成本变化而明显改变？

## 项目整体流程

```text
CSMAR 原始数据
    ↓
数据清洗与标准化
    ↓
建立历史股票池
    ↓
计算 PE / Momentum / ROE
    ↓
横截面排名与五分位分组
    ↓
因子有效性检验：Rank IC / 分组收益 / Q5-Q1
    ↓
构建单因子等权组合
    ↓
月度或季度历史回测
    ↓
加入交易成本
    ↓
风险收益评价
    ↓
稳健性测试
    ↓
研究图表与最终研究报告
```

## 当前项目结构

```text
a-share-factor-project/
│
├── data/
│   ├── raw/                    # CSMAR 原始数据
│   └── processed/              # 清洗后的标准数据
│
├── src/
│   ├── data_loader.py          # 读取和整理数据
│   ├── universe.py             # 建立历史股票池
│   ├── factors.py              # 计算基础因子
│   ├── factor_test.py          # 因子分组和 IC 研究
│   ├── portfolio.py            # 选择股票并构建组合
│   ├── backtest.py             # 执行历史回测
│   ├── performance.py          # 计算收益与风险指标
│   └── costs.py                # 计算交易成本
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_factor_research.ipynb
│   ├── 03_strategy.ipynb
│   └── 04_backtest_analysis.ipynb
│
├── reports/
│   ├── figures/                # 研究图表
│   └── final_report.md         # 最终研究报告
│
├── tests/
│   ├── test_data.py
│   ├── test_factors.py
│   └── test_backtest.py
│
├── configs/
│   └── config.yaml
│
├── README.md
├── requirements.txt
├── todo.md
└── run.py
```

## 当前开发状态

阶段 0～12 已完成：数据审计与标准化、历史股票池、因子计算与检验、目标组合、历史回测、交易成本、风险收益评价、稳健性测试、研究图表和最终报告均已有对应产物。PE 当日可用性仍是研究假设，不能据此将 PE 结果表述为已严格消除前视偏差。

## 本地数据再生

CSMAR 原始数据与 `data/processed/` 下的标准数据、股票池和审计产物均不纳入版本控制。准备好 `项目初始数据/` 后，按以下顺序在项目根目录执行：

```bash
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
python -m pytest -q
```

`run.py` 不带参数时只显示帮助，不会自动重跑任何阶段。阶段 11 生成 `reports/figures/` 和图表索引；最终报告读取这些已固化结果，不重新选择参数或改变研究规则。
