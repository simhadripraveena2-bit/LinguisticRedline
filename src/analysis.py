"""Statistical and qualitative analysis for LLM perceptions with real tract data."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import f_oneway
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression

DESCRIPTIONS_PATH = Path("data/neighborhood_descriptions.csv")
RESPONSES_PATH = Path("data/llm_responses.csv")
OUTPUT_DIR = Path("outputs")

THREAT_KEYWORDS = ["danger", "unsafe", "crime", "risk", "violence", "fear", "theft", "robbery", "assault"]
MIN_CATEGORY_COUNT = 10

def load_merged_data() -> pd.DataFrame:
    """Load and merge neighborhood descriptions with LLM outputs."""
    desc = pd.read_csv(DESCRIPTIONS_PATH)
    resp = pd.read_csv(RESPONSES_PATH)
    merged = desc.merge(resp, on="id", how="inner")
    return merged.dropna(subset=["numeric_score", "qualitative_response"]).copy()


def run_anova(merged: pd.DataFrame) -> pd.DataFrame:
    """Run one-way ANOVA for dominant race, income bucket, amenity bucket, and city."""
    factors = ["dominant_race", "income_bucket", "amenity_bucket", "city"]
    rows = []
    for factor in factors:
        groups = [grp["numeric_score"].values for _, grp in merged.groupby(factor)]
        if len(groups) < 2:
            continue
        f_stat, p_val = f_oneway(*groups)
        rows.append({"factor": factor, "f_stat": f_stat, "p_value": p_val})
    return pd.DataFrame(rows)

def warn_low_sample_categories(merged: pd.DataFrame, min_count: int = MIN_CATEGORY_COUNT) -> None:
    """Warn when dominant_race or income_bucket categories are too sparse for stable analysis."""
    for column in ["dominant_race", "income_bucket"]:
        counts = merged[column].value_counts(dropna=False)
        sparse = counts[counts < min_count]
        if sparse.empty:
            continue
        print(f"[warning] Low-sample categories in {column} (<{min_count} tracts):")
        for category, count in sparse.items():
            print(f"  - {category}: {count}")

def run_regression(merged: pd.DataFrame) -> pd.DataFrame:
    """Fit linear regression with race percentages and contextual control variables."""
    model_df = merged[
        [
            "numeric_score",
            "pct_black",
            "pct_white",
            "pct_hispanic",
            "pct_asian",
            "vacancy_rate",
            "income_bucket",
            "amenity_bucket",
            "city",
            "dominant_race",
        ]
    ].copy()

    X_numeric = model_df[["pct_black", "pct_white", "pct_hispanic", "pct_asian", "vacancy_rate"]].fillna(0)
    X_cat = pd.get_dummies(
        model_df[["income_bucket", "amenity_bucket", "city", "dominant_race"]],
        drop_first=True,
    )
    X = pd.concat([X_numeric, X_cat], axis=1)
    y = model_df["numeric_score"]

    reg = LinearRegression()
    reg.fit(X, y)

    return pd.DataFrame({"feature": X.columns, "coefficient": reg.coef_}).sort_values(
        "coefficient", key=np.abs, ascending=False
    )


def city_breakdown(merged: pd.DataFrame) -> pd.DataFrame:
    """Compute city-level summary statistics of risk and response volume."""
    summary = merged.groupby("city").agg(
        mean_numeric_score=("numeric_score", "mean"),
        std_numeric_score=("numeric_score", "std"),
        tracts=("id", "count"),
        mean_vacancy_rate=("vacancy_rate", "mean"),
    )
    return summary.reset_index().sort_values("mean_numeric_score", ascending=False)


def city_race_breakdown(merged: pd.DataFrame) -> pd.DataFrame:
    """Summarize mean risk by city and dominant race to assess cross-city differences."""
    out = (
        merged.groupby(["city", "dominant_race"])["numeric_score"]
        .mean()
        .reset_index(name="mean_numeric_score")
        .sort_values(["city", "mean_numeric_score"], ascending=[True, False])
    )
    return out


def threat_keyword_counts(merged: pd.DataFrame) -> pd.DataFrame:
    """Count threat-coded keyword usage in qualitative responses by dominant race."""
    vectorizer = CountVectorizer(vocabulary=THREAT_KEYWORDS, lowercase=True)
    matrix = vectorizer.fit_transform(merged["qualitative_response"].astype(str))
    counts = pd.DataFrame(matrix.toarray(), columns=vectorizer.get_feature_names_out())
    counts["dominant_race"] = merged["dominant_race"].values
    grouped = counts.groupby("dominant_race").sum().reset_index()
    return grouped.melt(id_vars="dominant_race", var_name="keyword", value_name="count")


def generate_plots(merged: pd.DataFrame, city_df: pd.DataFrame) -> None:
    """Generate analysis visualizations and save them to disk."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(11, 6))
    sns.boxplot(data=merged, x="dominant_race", y="numeric_score", hue="income_bucket")
    plt.xticks(rotation=20)
    plt.title("Crime Risk by Dominant Race and Income Bucket")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "boxplot_dominant_race_income_bucket.png")
    plt.close()

    pivot = merged.pivot_table(index="dominant_race", columns="city", values="numeric_score", aggfunc="mean")
    plt.figure(figsize=(14, 6))
    sns.heatmap(pivot, cmap="mako", annot=False)
    plt.title("Mean LLM Crime Risk by Dominant Race Across Cities")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "heatmap_race_city_scores.png")
    plt.close()

    plt.figure(figsize=(12, 5))
    sns.barplot(data=city_df, x="city", y="mean_numeric_score")
    plt.xticks(rotation=35, ha="right")
    plt.title("City-level Mean LLM Crime Risk")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "city_mean_scores.png")
    plt.close()


def main() -> None:
    """Run complete analysis workflow and save tabular + plot outputs."""
    merged = load_merged_data()
    warn_low_sample_categories(merged)
    anova_df = run_anova(merged)
    reg_df = run_regression(merged)
    city_df = city_breakdown(merged)
    city_race_df = city_race_breakdown(merged)
    keywords_df = threat_keyword_counts(merged)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT_DIR / "merged_with_scores.csv", index=False)
    anova_df.to_csv(OUTPUT_DIR / "anova_results.csv", index=False)
    reg_df.to_csv(OUTPUT_DIR / "regression_coefficients.csv", index=False)
    city_df.to_csv(OUTPUT_DIR / "city_breakdown.csv", index=False)
    city_race_df.to_csv(OUTPUT_DIR / "city_race_breakdown.csv", index=False)
    keywords_df.to_csv(OUTPUT_DIR / "threat_keyword_counts.csv", index=False)

    generate_plots(merged, city_df)
    print("Analysis complete. Outputs written to outputs/")


if __name__ == "__main__":
    main()
