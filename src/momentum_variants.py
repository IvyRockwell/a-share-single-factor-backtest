"""Run the 12-1 / 6-1 / 1M momentum variants side by side."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.factor_test import (
    FactorTestConfig,
    build_factor_test_details,
    build_ic_series,
    build_quantile_returns,
    build_summary,
    load_factor_test_config,
)
from src.factors import FactorConfig, load_factor_config


@dataclass(frozen=True)
class MomentumVariantOutputs:
    summary: pd.DataFrame
    ic: pd.DataFrame
    quantiles: pd.DataFrame


def load_variant_config(config_path: Path) -> tuple[FactorConfig, FactorTestConfig]:
    return load_factor_config(config_path), load_factor_test_config(config_path)


def build_variant_outputs(factor_details: pd.DataFrame, forward_returns: pd.DataFrame,
                          factor_config: FactorConfig, test_config: FactorTestConfig) -> MomentumVariantOutputs:
    """Build IC, quantile and summary tables for the primary and variant definitions."""
    variants = ("momentum",) + tuple(variant.name for variant in factor_config.momentum_variants)
    details = build_factor_test_details(
        factor_details,
        forward_returns,
        test_config.quantile_groups,
        score_type="raw",
        score_suffix="_score",
        eligibility_suffix="_is_rankable",
        factors=variants,
    )
    _, quantiles = build_quantile_returns(details, test_config.quantile_groups)
    ic = build_ic_series(details, test_config.min_ic_pairs)
    summary = build_summary(details, quantiles, ic)
    return MomentumVariantOutputs(summary, ic, quantiles)


def render_report(outputs: MomentumVariantOutputs) -> str:
    lines = [
        "# Momentum 变体：12-1、6-1、1M 反转",
        "",
        "主组合使用 12-1；6-1 与 1M 反转仅作横截面检验，不进入主组合。",
        "1M 反转的排序方向相反：过去一个月收益越低，分数越高。",
        "",
        "| 因子 | 有效 IC 期数 | 平均 Rank IC | IC 胜率 | IC t（NW） | Q5-Q1 月均 | Q5-Q1 t（NW） |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in outputs.summary.sort_values("factor").itertuples(index=False):
        lines.append(
            f"| {row.factor} | {row.ic_valid_periods} | {row.ic_mean:.4f} | {row.ic_win_rate:.2%} | "
            f"{row.ic_t_nw:.2f} | {row.spread_mean:.4%} | {row.spread_t_nw:.2f} |"
        )
    lines.extend([
        "",
        "## 口径",
        "",
        "- 12-1：信号日前 12 个月到前 1 个月；6-1：前 6 个月到前 1 个月；1M：前 1 个月到信号日，方向为反转。",
        "- 所有价格使用 `Adjprcwd`，起始/结束价格都取目标日或之前最近的有效交易日。",
        "- 12-1 的样本比 6-1/1M 少前 12 个月，报告按有效 IC 期数而非日历期数比较。",
        "",
    ])
    return "\n".join(lines)


def run(factor_config: FactorConfig, test_config: FactorTestConfig) -> MomentumVariantOutputs:
    directory = factor_config.processed_dir
    factor_details = pd.read_parquet(directory / "factor_details.parquet")
    forward = pd.read_parquet(directory / "factor_forward_returns.parquet")
    outputs = build_variant_outputs(factor_details, forward, factor_config, test_config)
    outputs.summary.to_csv(directory / "momentum_variants_summary.csv", index=False, encoding="utf-8-sig")
    outputs.ic.to_csv(directory / "momentum_variants_ic.csv", index=False, encoding="utf-8-sig")
    outputs.quantiles.to_csv(directory / "momentum_variants_quantiles.csv", index=False, encoding="utf-8-sig")
    report_dir = directory.parent.parent / "reports" / "generated"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "momentum-variants.md").write_text(render_report(outputs), encoding="utf-8")
    print(f"Built {len(outputs.summary)} momentum variants.")
    return outputs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run momentum 12-1 / 6-1 / 1M variants.")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    args = parser.parse_args(argv)
    factor_config, test_config = load_variant_config(args.config)
    run(factor_config, test_config)


if __name__ == "__main__":
    main()
