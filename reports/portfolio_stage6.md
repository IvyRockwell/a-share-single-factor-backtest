# 阶段 6：单因子目标组合与执行审计

## 固定规则

- 每个因子独立在每个信号日从可排名股票中选择 Top 20%；边界同分全部保留，不按股票代码拆分。
- 目标持仓等权，目标权重在每个因子和信号日严格加总为 1。
- 执行日为信号日后的首个市场交易日，执行价格口径为开盘；可执行需有该日交易记录、`trading_status=1` 且开盘价有效。
- 继续入选股票不视为无委托；阶段 7 在每个执行日按开盘前组合净值重新评估等权调整。
- 不可买入不替代、不跨日追单；完全退出但不可卖出的持仓由阶段 7 逐日重试。
- 2025-12-31 的目标候选保留，但因数据末端没有后续执行日而不标记为已执行。

## 输出核对

- 目标持仓：275,186 条；调仓动作：326,897 条；逐期汇总：402 条。
- 各期目标权重均为 1；不可买入对应的目标现金权重总和为 0.133329。

## 固定抽样复核

| 因子 | 信号日 | 股票 | 因子分数 | 入选排名百分位 | 目标权重 | 执行日 | 执行状态 | 原因 |
| --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |
| momentum | 2019-02-28 | 601318 | 0.876391 | 0.876391 | 0.00146413 | 2019-03-01 | rebalance_required | target_continues_and_is_rebalanced_in_stage7 |
| momentum | 2021-09-30 | 600132 | 0.976060 | 0.976060 | 0.00124688 | 2021-10-08 | rebalance_required | target_continues_and_is_rebalanced_in_stage7 |
| momentum | 2023-11-30 | 600496 | 0.893472 | 0.893472 | 0.00103627 | 2023-12-01 | executable | normal_trading_status_and_valid_open_price |
| momentum | 2024-09-30 | 300628 | 0.975213 | 0.975213 | 0.00101523 | 2024-10-08 | rebalance_required | target_continues_and_is_rebalanced_in_stage7 |
| pe | 2016-12-30 | 300118 | 0.919577 | 0.919577 | 0.00229358 | 2017-01-03 | rebalance_required | target_continues_and_is_rebalanced_in_stage7 |
| pe | 2019-12-31 | 600567 | 0.961336 | 0.961336 | 0.00175747 | 2020-01-02 | rebalance_required | target_continues_and_is_rebalanced_in_stage7 |
| pe | 2024-01-31 | 000935 | 0.857888 | 0.857888 | 0.00130378 | 2024-02-01 | executable | normal_trading_status_and_valid_open_price |
| pe | 2024-01-31 | 002521 | 0.801304 | 0.801304 | 0.00130378 | 2024-02-01 | executable | normal_trading_status_and_valid_open_price |

## 限制

- 本阶段只形成目标组合和执行审计，不计算实际市值权重、现金收益、组合收益或延迟卖出后的成交结果。
- 现有可靠字段不足以识别涨跌停造成的拒单；该限制不被静默视为可成交。
