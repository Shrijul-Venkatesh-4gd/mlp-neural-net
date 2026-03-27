from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

from app.data.data_loader import AdultDataset


def _code_block(text: str) -> str:
    return f"```text\n{text.rstrip()}\n```"


def _section(title: str, body: str) -> str:
    return f"## {title}\n{body.strip()}\n"


def _bullet_lines(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def _rate_table(df: pd.DataFrame, column: str, top_n: int = 8) -> pd.DataFrame:
    grouped = (
        df.groupby(column, dropna=False)["income"]
        .agg(
            count="size",
            high_income_rate=lambda series: (series.eq(">50K").mean() * 100),
        )
        .sort_values(["count", "high_income_rate"], ascending=[False, False])
        .head(top_n)
    )
    grouped["high_income_rate"] = grouped["high_income_rate"].round(2)
    return grouped


def build_eda_report() -> str:
    dataset = AdultDataset.load()
    features = dataset.get_features()
    target = dataset.get_target(normalized=True)
    df = dataset.to_frame(normalized_target=True)

    question_marks = {
        column: int(features[column].eq("?").sum())
        for column in features.select_dtypes(exclude=["number"]).columns
    }
    missing_summary = pd.DataFrame(
        {
            "nan_count": features.isna().sum(),
            "question_mark_count": pd.Series(question_marks),
        }
    ).fillna(0).astype(int)
    missing_summary["total_missing_like"] = (
        missing_summary["nan_count"] + missing_summary["question_mark_count"]
    )
    missing_summary = missing_summary.sort_values(
        ["total_missing_like", "nan_count"], ascending=False
    )

    numeric_summary = features.describe(include=["number"]).round(2)

    target_distribution = pd.DataFrame(
        {
            "count": target.value_counts(),
            "percentage": (target.value_counts(normalize=True) * 100).round(2),
        }
    )

    target_numeric_means = (
        df.groupby("income")[
            [
                "age",
                "education-num",
                "capital-gain",
                "capital-loss",
                "hours-per-week",
            ]
        ]
        .mean()
        .round(2)
    )

    correlations = (
        features.select_dtypes(include=["number"])
        .assign(target=target.map({"<=50K": 0, ">50K": 1}))
        .corr(numeric_only=True)["target"]
        .drop("target")
        .sort_values(ascending=False)
        .round(3)
    )

    zero_inflation = {
        "capital-gain": round(float(features["capital-gain"].eq(0).mean() * 100), 2),
        "capital-loss": round(float(features["capital-loss"].eq(0).mean() * 100), 2),
    }

    cardinality = (
        features.select_dtypes(exclude=["number"])
        .nunique(dropna=False)
        .sort_values(ascending=False)
    )

    top_categories = []
    for column in ["workclass", "education", "occupation", "native-country"]:
        counts = features[column].value_counts(dropna=False).head(8)
        top_categories.append(f"### {column}")
        top_categories.append(_code_block(counts.to_string()))

    sections = [
        "# Adult Dataset EDA Report",
        _section(
            "Overview",
            _bullet_lines(
                [
                    f"Rows: {features.shape[0]}",
                    f"Features: {features.shape[1]}",
                    f"Target: {target.name or 'income'}",
                    f"Duplicate feature rows: {int(features.duplicated().sum())}",
                    "Task type: binary income classification",
                    "Source: UCI Adult / Census Income dataset",
                ]
            ),
        ),
        _section(
            "How To Read This Dataset",
            _bullet_lines(
                [
                    "Each row represents one person from the census-derived sample.",
                    "The target is whether annual income is above `50K`.",
                    "`education-num` is an ordinal encoding of `education`, so the two columns carry overlapping information.",
                    "`capital-gain` and `capital-loss` are extremely sparse and should be treated as zero-inflated features.",
                    "Some categorical fields use `?` as a missing-like placeholder in addition to actual `NaN` values.",
                ]
            ),
        ),
        _section(
            "Target Distribution",
            _code_block(target_distribution.to_string()),
        ),
        _section(
            "Data Quality",
            "\n".join(
                [
                    _bullet_lines(
                        [
                            "The target is imbalanced: most rows belong to the `<=50K` class.",
                            "There are both real `NaN` values and literal `?` markers in categorical columns.",
                            "A small number of duplicate feature rows exist and should be reviewed before training.",
                        ]
                    ),
                    "",
                    _code_block(missing_summary.to_string()),
                ]
            ),
        ),
        _section(
            "Numeric Feature Summary",
            _code_block(numeric_summary.to_string()),
        ),
        _section(
            "Categorical Cardinality",
            _code_block(cardinality.to_string()),
        ),
        _section(
            "Most Common Categories",
            "\n".join(top_categories),
        ),
        _section(
            "Income Differences By Target",
            _code_block(target_numeric_means.to_string()),
        ),
        _section(
            "Strongest Numeric Signals",
            "\n".join(
                [
                    _code_block(correlations.to_string()),
                    "",
                    _bullet_lines(
                        [
                            f"`capital-gain` is zero in {zero_inflation['capital-gain']}% of rows.",
                            f"`capital-loss` is zero in {zero_inflation['capital-loss']}% of rows.",
                            "Higher income is most associated with education level, age, and hours worked per week among the numeric columns.",
                            "`fnlwgt` appears to have almost no linear relationship with the target.",
                        ]
                    ),
                ]
            ),
        ),
        _section(
            "High-Income Rate By Key Categories",
            "\n".join(
                [
                    "### sex",
                    _code_block(_rate_table(df, "sex").to_string()),
                    "### workclass",
                    _code_block(_rate_table(df, "workclass").to_string()),
                    "### education",
                    _code_block(_rate_table(df, "education").to_string()),
                    "### marital-status",
                    _code_block(_rate_table(df, "marital-status").to_string()),
                ]
            ),
        ),
        _section(
            "Recommended Report Extensions",
            _bullet_lines(
                [
                    "Add plots for age, hours-per-week, and education level split by income class.",
                    "Show target rate by category for `education`, `occupation`, `workclass`, and `marital-status` to expose useful nonlinear patterns.",
                    "Separate true missing values from `?` placeholders, because this dataset uses both.",
                    "Add outlier notes for `capital-gain`, `capital-loss`, and `hours-per-week` because these features are heavily skewed.",
                    "Document feature redundancy between `education` and `education-num` so downstream models do not double-count the same signal.",
                    "Include fairness-oriented slices for `sex` and `race` if the report is meant to support modeling decisions.",
                    "Add a preprocessing section describing imputation, encoding, duplicate handling, and class-imbalance strategy.",
                ]
            ),
        ),
    ]

    return "\n\n".join(section.rstrip() for section in sections).strip() + "\n"


def write_eda_report(output_path: str = "docs/data/eda_report.md") -> Path:
    report = build_eda_report()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(report, encoding="utf-8")
    return destination


def main() -> None:
    parser = ArgumentParser(description="Generate a markdown EDA report for the Adult dataset.")
    parser.add_argument(
        "--output",
        default="docs/data/eda_report.md",
        help="Where to write the markdown report.",
    )
    args = parser.parse_args()

    output = write_eda_report(args.output)
    print(f"Wrote EDA report to {output}")


if __name__ == "__main__":
    main()
