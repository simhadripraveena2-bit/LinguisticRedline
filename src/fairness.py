"""Compute fairness metrics over LLM crime-risk score outputs with city and vacancy controls."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DESCRIPTIONS_PATH = Path("data/neighborhood_descriptions.csv")
RESPONSES_PATH = Path("data/llm_responses.csv")
OUTPUT_PATH = Path("outputs/fairness_report.csv")
MIN_GROUP_SIZE = 10


def load_merged_data() -> pd.DataFrame:
    """Load and merge tract attributes with LLM numeric score outputs."""
    desc = pd.read_csv(DESCRIPTIONS_PATH)
    resp = pd.read_csv(RESPONSES_PATH)
    return desc.merge(resp, on="id", how="inner").dropna(subset=["numeric_score"]).copy()


def disparate_impact_ratio(
    merged: pd.DataFrame,
    high_risk_threshold: float = 7.0,
    vacancy_quantile: float = 0.5,
) -> pd.DataFrame:
    """Compute disparate impact ratio by dominant race within low/high vacancy strata."""
    data = merged.copy()
    split = data["vacancy_rate"].quantile(vacancy_quantile)
    data["vacancy_band"] = data["vacancy_rate"].apply(lambda x: "low_vacancy" if x <= split else "high_vacancy")
    data["high_risk"] = data["numeric_score"] >= high_risk_threshold

    rows = []
    for band, band_df in data.groupby("vacancy_band"):
        counts = band_df["dominant_race"].value_counts(dropna=False)
        eligible_groups = counts[counts >= MIN_GROUP_SIZE].index
        skipped_groups = counts[counts < MIN_GROUP_SIZE]
        for race, count in skipped_groups.items():
            print(
                f"[warning] Skipping disparate impact for dominant_race={race} in {band} "
                f"(only {count} tracts; minimum {MIN_GROUP_SIZE})."
            )

        rates = band_df[band_df["dominant_race"].isin(eligible_groups)].groupby("dominant_race")["high_risk"].mean()
        if rates.empty:
            print(f"[warning] No eligible dominant_race groups for disparate impact in {band}.")
            continue
        ref_rate = rates.max()
        for race, rate in rates.items():
            ratio = rate / ref_rate if ref_rate > 0 else 0.0
            rows.append(
                {
                    "vacancy_band": band,
                    "dominant_race": race,
                    "high_risk_rate": rate,
                    "disparate_impact_ratio": ratio,
                }
            )
    return pd.DataFrame(rows)


def demographic_parity_gap(merged: pd.DataFrame) -> pd.DataFrame:
    """Compute demographic parity gap in mean score by dominant race."""
    global_mean = merged["numeric_score"].mean()
    counts = merged["dominant_race"].value_counts(dropna=False)
    skipped_groups = counts[counts < MIN_GROUP_SIZE]
    for race, count in skipped_groups.items():
        print(
            f"[warning] Skipping demographic parity for dominant_race={race} "
            f"(only {count} tracts; minimum {MIN_GROUP_SIZE})."
        )
    eligible_groups = counts[counts >= MIN_GROUP_SIZE].index
    means = merged[merged["dominant_race"].isin(eligible_groups)].groupby("dominant_race")["numeric_score"].mean()
    out = means.reset_index(name="mean_score")
    out["demographic_parity_gap"] = out["mean_score"] - global_mean
    return out


def city_fairness_breakdown(merged: pd.DataFrame) -> pd.DataFrame:
    """Compute city-level fairness summaries by dominant race."""
    counts = merged.groupby(["city", "dominant_race"]).size().rename("count").reset_index()
    sparse = counts[counts["count"] < MIN_GROUP_SIZE]
    for row in sparse.itertuples(index=False):
        print(
            f"[warning] Skipping city fairness for city={row.city}, dominant_race={row.dominant_race} "
            f"(only {row.count} tracts; minimum {MIN_GROUP_SIZE})."
        )

    eligible = counts[counts["count"] >= MIN_GROUP_SIZE][["city", "dominant_race"]]
    filtered = merged.merge(eligible, on=["city", "dominant_race"], how="inner")

    group_means = (
        filtered.groupby(["city", "dominant_race"])["numeric_score"]
        .mean()
        .reset_index(name="mean_score")
    )
    city_global = merged.groupby("city")["numeric_score"].mean().rename("city_mean")
    out = group_means.merge(city_global, on="city", how="left")
    out["city_demographic_parity_gap"] = out["mean_score"] - out["city_mean"]
    return out.sort_values(["city", "city_demographic_parity_gap"], ascending=[True, False])


def strongest_predictors(merged: pd.DataFrame, top_n: int = 12) -> pd.DataFrame:
    """Identify highest-score groups across key categorical dimensions."""
    factors = ["dominant_race", "income_bucket", "amenity_bucket", "city"]
    grouped = []
    for factor in factors:
        tmp = merged.groupby(factor)["numeric_score"].mean().reset_index()
        tmp.columns = ["factor_level", "mean_numeric_score"]
        tmp["factor"] = factor
        grouped.append(tmp)
    all_levels = pd.concat(grouped, ignore_index=True)
    return all_levels.sort_values("mean_numeric_score", ascending=False).head(top_n)


def main() -> None:
    """Run fairness analyses and save outputs to the outputs directory."""
    merged = load_merged_data()
    di = disparate_impact_ratio(merged)
    dp = demographic_parity_gap(merged)
    city = city_fairness_breakdown(merged)
    pred = strongest_predictors(merged)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    di.to_csv(OUTPUT_PATH.with_name("disparate_impact_by_vacancy.csv"), index=False)
    dp.to_csv(OUTPUT_PATH, index=False)
    city.to_csv(OUTPUT_PATH.with_name("fairness_by_city.csv"), index=False)
    pred.to_csv(OUTPUT_PATH.with_name("strongest_predictors.csv"), index=False)

    print(f"Saved fairness outputs to {OUTPUT_PATH.parent}")


if __name__ == "__main__":
    main()