# A 股单因子研究：PE、Momentum、ROE

[![tests](https://github.com/IvyRockwell/a-share-single-factor-backtest/actions/workflows/tests.yml/badge.svg)](https://github.com/IvyRockwell/a-share-single-factor-backtest/actions/workflows/tests.yml)

用 CSMAR A 股数据检验价值（PE）、动量（Momentum）、质量（ROE）三个基础因子的选股能力，并把横截面结论落到一个按开盘成交、计入佣金与印花税、等权月度调仓的回测上。样本 2014-09-12 至 2025-12-31，136 个月末信号日，2,735 个交易日。

**研究结论见 [`reports/research_note.md`](reports/research_note.md)。**

## 主要结果

| 因子 | 平均 Rank IC | IC 胜率 | 累计 Q5-Q1 | Top 20% 毛累计 | 年化 | 最大回撤 | Sharpe |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PE | 0.0396 | 62.96% | 36.17% | 226.56% | 11.52% | -46.37% | 0.592 |
| Momentum | -0.0236 | 39.53% | -74.23% | 25.76% | 2.13% | -62.32% | 0.211 |
| ROE | 0.0065 | 48.89% | -43.26% | 117.99% | 7.44% | -53.71% | 0.419 |
| H00300 | — | — | — | 141.94% | 8.48% | -46.06% | 0.483 |
| 等权全 A | — | — | — | 193.90% | 10.44% | -61.91% | 0.486 |
| 市值加权全 A | — | — | — | 143.30% | 8.54% | -48.43% | 0.476 |

只有 PE 的分组、IC 与回测方向一致；Momentum 与预设方向相反，ROE 的横截面区分度接近零。加入 20 bp 滑点后三者净累计收益分别为 200.24%、-11.77%、105.49%。

**超额里有多少是因子？** 组合是全 A 等权，市值加权全 A 与 HS300 几乎重合（8.54% vs 8.48%），而等权全 A 是 10.44%。所以 PE 相对 HS300 的 3.04 个百分点里约 1.96 是等权/小市值暴露，因子本身约 1.08。对行业与市值做中性化后，PE 的 IC 从 0.0396 **升到 0.0464**，ICIR 从 0.24 **升到 0.48**——低估值信号不是小盘股的伪装。四个子区间里 PE 的中性化 IC 全部为正（NW t 1.8–4.2，其中 2020–2022 不显著），Momentum 三段为负，见研究笔记 4.3 节。

![三因子毛净值与 H00300](reports/figures/05_gross_nav_vs_benchmark.png)

![中性化前后平均 Rank IC](reports/figures/14_neutralization_ic.png)

## 数据

CSMAR（国泰安）。原始导出需自行获取，放入项目根目录的 `项目初始数据/`，**不随仓库分发**——CSMAR 数据受其最终用户许可协议约束，版权归深圳希施玛数据科技有限公司。

| 表 | 用途 |
| --- | --- |
| `TRD_Dalyr` | 个股日行情（收益、复权收盘价、交易状态） |
| `STK_MKT_ValuationMetrics` | 估值指标（`PE1TTM`） |
| `FS_Comins` / `FS_Combas` / `IAR_Forecdt` | 利润表、资产负债表、实际披露日 |
| `TRD_Co` / `STK_LISTEDCOINFOANL` / `STK_ITEMCHANGE` | 上市日、上市状态变更 |
| `STK_CAPITAL` | 股本变动（总股数、流通股数） |
| `STK_INDUSTRYCLASS` | 行业分类（证监会 2012 版） |
| `TRD_Cale` | 交易日历 |
| `IDX_Idxtrd` | 沪深 300 全收益（H00300） |

## 方法要点

- **股票池**：月末最后一个交易日构建，要求当日正常交易、上市满 6 个自然月、未处于暂停上市 / 退市整理 / 终止上市。状态按事件生效日判断，后来退市的股票在退市前仍留在池中，避免幸存者偏差。
- **因子**：PE 用按实际披露日重建的 TTM 净利润 ÷ 总市值，为正者反向排名（供应商 `PE1TTM` 只留作对照，实测含前视）；Momentum 取不晚于 6 个自然月前的复权收盘价算区间收益；ROE 只取信号日前已实际披露的合并年报。
- **中性化**：每个信号日对 `log(市值)` 与证监会行业哑变量做 OLS，取残差重新排名；样本不足时自动降级为纯市值。
- **基准**：H00300、等权全 A、市值加权全 A 三条，用来分离"基准错配"与"因子本身"。
- **检验**：五分位分组、下月收益、Spearman Rank IC、Q5-Q1；t 值用 Newey-West 修正，并按四个风格阶段给出子区间稳定性。
- **组合**：Top 20% 等权、月度调仓、下一交易日开盘执行；先退出与减仓，再新建与加仓，现金不足时按比例缩减。
- **成本**：佣金双边 3 bp，印花税卖出 10 bp（2023-08-28 起 5 bp），滑点 0 / 10 / 20 bp 三档情景。

## 目录

```text
configs/config.yaml     全部参数与研究假设
src/data_loader.py      原始 CSV → 标准化 parquet，附质量标记
src/universe.py         历史股票池（point-in-time）
src/market_cap.py       总股本 × 收盘价，逐月市值
src/factors.py          PE / Momentum / ROE + 行业市值中性化
src/factor_test.py      分组、未来收益、Rank IC 与 t 值
src/portfolio.py        目标组合与调仓动作
src/backtest.py         开盘成交、等权恢复、现金约束的逐日账本
src/costs.py            佣金、印花税、滑点
src/performance.py      收益、波动、回撤、Sharpe，三条基准
src/robustness.py       参数与成本网格
src/visualization.py    研究图表
reports/research_note.md   研究笔记（定稿）
reports/figures/           18 张图
notebooks/01_data_exploration.ipynb  原始数据探查
tests/                     74 个用例（构造数据，不依赖原始数据）
run.py                     命令行入口
```

## 运行

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python run.py clean universe marketcap factors evaluate portfolio backtest costs performance robustness figures
python run.py all          # 等价于按上面的顺序全部执行
python -m pytest -q        # 不需要原始数据
```

各步骤把审计报告写入 `reports/generated/`（不纳入版本控制），`reports/research_note.md` 与 `reports/figures/` 为定稿产物。图表中文标签使用 Microsoft YaHei / SimHei，在 macOS 或 Linux 上重跑 `figures` 需要改 `src/visualization.py` 中的 `font.sans-serif`。

## 口径与已知限制

- PE 由披露日重建，供应商 `PE1TTM` 实测含前视：季度末净利润在披露前就被用于计算，已披露月份两者逐位一致（见研究笔记 5.3）。
- 中性化用"总股本 × 收盘价"的市值，未做自由流通市值调整；行业为证监会 2012 版大类，按实施日逐月对齐。
- 信号日按 `Trdsta=1` 已排除 ST/*ST，但持仓期内转为 ST 或长期停牌的不做单独处理；次新股门槛 6 个自然月；|日收益| > 100% 的记录保留不 winsorize。
- 涨跌停、整手、最低佣金、盘口深度、资金容量未建模；现金收益为 0。
- IC 与多空 t 值已按 Newey-West 修正，并给出四个子区间结果，但仍是同一样本、事后选定三个因子，没有样本外区间，也没有多重检验修正。
- Sharpe 按 252 个交易日年化、无风险利率取 0，分子为日算术均值，表中年化收益为几何复利，两者口径不同。
- 历史回测不能证明因果关系，也不代表未来收益。

## 许可

代码与文档 MIT；CSMAR 数据不在许可范围内。本项目仅用于研究，不构成投资建议。
