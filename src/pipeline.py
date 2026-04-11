
"""Master runner for LinguisticRedline real-data pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from fetch_census import main as fetch_census_main
from fetch_osm import main as fetch_osm_main
from generate_descriptions import main as generate_descriptions_main

CENSUS_PATH = Path("data/census_tracts.csv")
AMENITY_PATH = Path("data/tracts_with_amenities.csv")
DESCRIPTIONS_PATH = Path("data/neighborhood_descriptions.csv")


def run_step_if_needed(path: Path, step_name: str, func, step_args: Optional[Sequence[str]] = None) -> None:
    """Run a pipeline step only when its expected output does not already exist."""
    if path.exists():
        print(f"[skip] {step_name}: {path} already exists")
        return
    print(f"[run] {step_name}")
    if step_args is None:
        func()
    else:
        func(step_args)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse pipeline flags for selective step execution behavior."""
    parser = argparse.ArgumentParser(description="Run full real-data pipeline")
    parser.add_argument("--limit", type=int, default=None, help="Only process first N tracts during fetch_osm")
    parser.add_argument("--resume", action="store_true", help="Resume fetch_osm from cached tract amenity JSON files")
    return parser.parse_args(argv)


def print_summary() -> None:
    """Print aggregate summary metrics for generated tract datasets."""
    census_df = pd.read_csv(CENSUS_PATH) if CENSUS_PATH.exists() else pd.DataFrame()
    amenity_df = pd.read_csv(AMENITY_PATH) if AMENITY_PATH.exists() else pd.DataFrame()

    print("\nPipeline Summary")
    print("----------------")
    print(f"Total tracts fetched: {len(census_df):,}")
    print(f"Tracts after filtering: {len(census_df):,}")
    print("Expected sample size: ~2,000 tracts (about 200 per city)")

    if not census_df.empty:
        print("\nDominant race distribution:")
        print(census_df["dominant_race"].value_counts(dropna=False).to_string())
        print("\nIncome bucket distribution:")
        print(census_df["income_bucket"].value_counts(dropna=False).to_string())

    if not amenity_df.empty and "amenity_bucket" in amenity_df.columns:
        print("\nAmenity bucket distribution:")
        print(amenity_df["amenity_bucket"].value_counts(dropna=False).to_string())


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Execute full real-data pipeline in order with skip-on-existing semantics."""
    args = parse_args(argv)
    fetch_osm_args = []
    if args.limit is not None:
        fetch_osm_args.extend(["--limit", str(args.limit)])
    if args.resume:
        fetch_osm_args.append("--resume")

    run_step_if_needed(CENSUS_PATH, "fetch_census", fetch_census_main)
    run_step_if_needed(AMENITY_PATH, "fetch_osm", fetch_osm_main, fetch_osm_args)
    run_step_if_needed(DESCRIPTIONS_PATH, "generate_descriptions", generate_descriptions_main)
    print_summary()


if __name__ == "__main__":
    main()
