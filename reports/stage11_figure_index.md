# 阶段 11：研究图表索引

本索引只记录阶段 11 图表。图表读取阶段 4～10 已固化产物，不改写收益、成本、回撤或稳健性规则。

## 统一视觉与口径

- Matplotlib 白底；PE `#0072B2`、Momentum `#E69F00`、ROE `#009E73`、H00300 `#4D4D4D`。
- gross 使用实线；net 0/10/20 bp 分别使用由浅到深颜色及 `--` / `-.` / `:` 线型。
- 中文字体为 Microsoft YaHei（后备 SimHei）；所有收益、回撤、成本与换手率均以百分比或明确单位显示。

## 逐图索引

| 文件名 | 回答的问题 | 数据来源及字段 | gross/net 与成本口径 | 日期范围 | 已知限制 |
| --- | --- | --- | --- | --- | --- |
| `01_sample_quality.png` | 股票池规模与三因子可排名样本如何变化？ | `factor_period_quality.csv: signal_date, factor, pool_count, rankable_count, unrankable_count, unrankable_by_reason` | 不适用 | 2014-09-30—2025-12-31 | Momentum 前 6 期无完整 6 个月历史价格；不插值。 |
| `02_factor_pe.png` | PE 的五分位、Q5-Q1 与 Rank IC 如何变化？ | `factor_quantile_returns.csv: quantile, cumulative_group_return, q_high_minus_q_low, cumulative_q_high_minus_q_low; factor_ic_series.csv: rank_ic` | 阶段 5 分组未来收益，非策略 gross/net；成本不适用 | 2014-09-30—2025-11-28 | 每期 Q5-Q1 只计算一次，累计值在同期五个分位行中完全一致。 |
| `03_factor_momentum.png` | Momentum 的五分位、Q5-Q1 与 Rank IC 如何变化？ | `同 02，factor=momentum` | 阶段 5 分组未来收益，非策略 gross/net；成本不适用 | 2015-03-31—2025-11-28 | 前 6 个信号期无完整回看历史，因此不填造结果。 |
| `04_factor_roe.png` | ROE 的五分位、Q5-Q1 与 Rank IC 如何变化？ | `同 02，factor=roe` | 阶段 5 分组未来收益，非策略 gross/net；成本不适用 | 2014-09-30—2025-11-28 | 阶段 5 结果不代表开盘可成交策略。 |
| `05_gross_nav_vs_benchmark.png` | 三因子毛净值相对 H00300 如何变化？ | `backtest_daily_returns.parquet: trade_date, factor, nav, benchmark_nav` | 策略 gross；H00300 全收益；未扣策略成本 | 2014-10-08—2025-12-31 | 策略执行日从开盘开始，基准使用完整日收益，盘中起点不完全一致。 |
| `06_gross_vs_net.png` | 交易成本情景如何改变各因子净值？ | `backtest_daily_returns.parquet: nav; backtest_stage8_daily_returns.parquet: net_nav, slippage_bp` | gross 与 net 0/10/20 bp；0 bp 仍含佣金和印花税 | 2014-10-08—2025-12-31 | 最低佣金与盘口冲击未纳入；滑点为研究假设。 |
| `07_drawdowns.png` | 各毛收益策略与 H00300 的历史回撤如何比较？ | `performance_drawdowns.parquet: series_type, factor, trade_date, drawdown` | 策略 gross；H00300 全收益；不展示 net | 2014-10-08—2025-12-31 | 策略与基准执行日盘中起点不完全一致。 |
| `08_monthly_returns.png` | 三因子毛收益策略每月表现如何？ | `performance_monthly_returns.csv: series_type, factor, month, monthly_return` | 仅 gross；未扣成本 | 2014-10—2025-12 | 统一色阶保留极端月份；缺失格显示为破折号。 |
| `09a_cumulative_return.png` | 累计收益在 gross、net 成本情景和基准间如何比较？ | `performance_metrics.csv: cumulative_return` | gross、net 0/10/20 bp、H00300 | 2014-10-08—2025-12-31 | H00300 不适用策略成本。 |
| `09b_annualized_return.png` | 年化收益在各序列间如何比较？ | `performance_metrics.csv: annualized_return` | 同 09a | 2014-10-08—2025-12-31 | 按 252 个交易日年化。 |
| `09c_annualized_volatility.png` | 年化波动率在各序列间如何比较？ | `performance_metrics.csv: annualized_volatility` | 同 09a | 2014-10-08—2025-12-31 | 日收益样本标准差按 252 个交易日年化。 |
| `09d_max_drawdown.png` | 最大回撤在各序列间如何比较？ | `performance_metrics.csv: max_drawdown` | 同 09a | 2014-10-08—2025-12-31 | 历史峰值包含初始净值 1。 |
| `09e_sharpe_ratio.png` | Sharpe 在各序列间如何比较？ | `performance_metrics.csv: sharpe_ratio` | 同 09a | 2014-10-08—2025-12-31 | 年化无风险利率为 0%，年化交易日为 252。 |
| `10_turnover_and_cost.png` | 因子换手率与累计交易成本有何差异？ | `performance_metrics.csv: annualized_turnover, cumulative_transaction_cost` | 换手率与滑点无关；成本为 net 0/10/20 bp | 2014-10-08—2025-12-31 | 累计成本为归一化组合的 NAV 点数，不等于期末毛净值减净净值。 |
| `11_robustness_pe.png` | PE 净累计收益对选股比例、调仓频率和成本是否敏感？ | `robustness_stage10_metrics.csv: selection_quantile, rebalance_frequency, slippage_bp, net_cumulative_return` | net 0/10/20 bp | 2014-10-08—2025-12-31 | 边界同分全部保留；不选择最佳参数。 |
| `12_robustness_momentum.png` | Momentum 净累计收益对参数是否敏感？ | `同 11，factor=momentum` | net 0/10/20 bp | 2014-10-08—2025-12-31 | 部分参数下收益方向反转；不选择最佳参数。 |
| `13_robustness_roe.png` | ROE 净累计收益对参数是否敏感？ | `同 11，factor=roe` | net 0/10/20 bp | 2014-10-08—2025-12-31 | 边界同分全部保留；不选择最佳参数。 |

## 固定研究限制

- H00300 为 CSMAR `Idxtrd08` 全收益；策略首次执行日从开盘开始，基准为完整日收益。
- 0 bp 仅表示无滑点，仍包含双边 3 bp 佣金和卖出印花税；最低佣金未纳入。
- PE1TTM 按交易日可用的既定假设处理，原始数据无法证明其财务分母已严格按披露时点消除前视偏差。
- 涨跌停无法可靠识别，正常交易状态和有效开盘价不保证实际成交。
