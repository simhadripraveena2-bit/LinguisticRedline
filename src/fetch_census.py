"""Fetch ACS tract-level demographic data for the 10 largest US cities."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List

import geopandas as gpd
import pandas as pd
import requests

from config_loader import load_config

OUTPUT_PATH = Path("data/census_tracts.csv")

ACS_VARIABLES = [
    "B02001_002E",  # White alone
    "B02001_003E",  # Black alone
    "B02001_004E",  # Native American
    "B02001_005E",  # Asian alone
    "B03001_003E",  # Hispanic
    "B01003_001E",  # Total population
    "B19013_001E",  # Median household income
    "B25002_003E",  # Vacant housing units
    "B25001_001E",  # Total housing units
    "B01002_001E",  # Median age
    "B15003_022E",  # Bachelor's holders
]

CITY_COUNTIES: Dict[str, List[Dict[str, str]]] = {
    "New York": [
        {"state": "36", "county": "005"},
        {"state": "36", "county": "047"},
        {"state": "36", "county": "061"},
        {"state": "36", "county": "081"},
        {"state": "36", "county": "085"},
    ],
    "Los Angeles": [{"state": "06", "county": "037"}],
    "Chicago": [{"state": "17", "county": "031"}],
    "Houston": [{"state": "48", "county": "201"}],
    "Phoenix": [{"state": "04", "county": "013"}],
    "Philadelphia": [{"state": "42", "county": "101"}],
    "San Antonio": [{"state": "48", "county": "029"}],
    "San Diego": [{"state": "06", "county": "073"}],
    "Dallas": [{"state": "48", "county": "113"}],
    "San Jose": [{"state": "06", "county": "085"}],
}


STATE_ABBR = {
    "01": "al", "02": "ak", "04": "az", "05": "ar", "06": "ca", "08": "co", "09": "ct", "10": "de",
    "11": "dc", "12": "fl", "13": "ga", "15": "hi", "16": "id", "17": "il", "18": "in", "19": "ia",
    "20": "ks", "21": "ky", "22": "la", "23": "me", "24": "md", "25": "ma", "26": "mi", "27": "mn",
    "28": "ms", "29": "mo", "30": "mt", "31": "ne", "32": "nv", "33": "nh", "34": "nj", "35": "nm",
    "36": "ny", "37": "nc", "38": "nd", "39": "oh", "40": "ok", "41": "or", "42": "pa", "44": "ri",
    "45": "sc", "46": "sd", "47": "tn", "48": "tx", "49": "ut", "50": "vt", "51": "va", "53": "wa",
    "54": "wv", "55": "wi", "56": "wy",
}


def request_with_retry(url: str, params: Dict[str, str], attempts: int = 3) -> List[List[str]]:
    """Request Census JSON payload with retries and exponential backoff."""
    for attempt in range(attempts):
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("Unreachable retry path")


def income_bucket(income: float) -> str:
    """Bucket a tract's median household income into ordered income categories."""
    if income < 35000:
        return "low"
    if income < 55000:
        return "lower_middle"
    if income < 80000:
        return "middle"
    if income < 110000:
        return "upper_middle"
    return "high"


def load_state_tract_geometries(state_fips: str, year: int) -> gpd.GeoDataFrame:
    """Load tract geometries for a state from TIGER shapefiles."""
    url = f"https://www2.census.gov/geo/tiger/TIGER{year}/TRACT/tl_{year}_{state_fips}_tract.zip"
    tracts = gpd.read_file(url)
    return tracts[["GEOID", "geometry"]]


def fetch_city_tracts(city: str, year: int, api_key: str) -> pd.DataFrame:
    """Fetch tract-level ACS5 data for all configured counties in a city."""
    base_url = f"https://api.census.gov/data/{year}/acs/acs5"
    all_frames: List[pd.DataFrame] = []

    for loc in CITY_COUNTIES[city]:
        params = {
            "get": ",".join(ACS_VARIABLES),
            "for": "tract:*",
            "in": f"state:{loc['state']} county:{loc['county']}",
            "key": api_key,
        }
        raw = request_with_retry(base_url, params)
        city_df = pd.DataFrame(raw[1:], columns=raw[0])
        city_df["city"] = city
        all_frames.append(city_df)

    combined = pd.concat(all_frames, ignore_index=True)
    return combined


def transform_census_data(df: pd.DataFrame, min_population: int) -> pd.DataFrame:
    """Convert ACS fields to numeric values and compute derived columns."""
    rename_map = {
        "B02001_002E": "white_pop",
        "B02001_003E": "black_pop",
        "B02001_004E": "native_pop",
        "B02001_005E": "asian_pop",
        "B03001_003E": "hispanic_pop",
        "B01003_001E": "total_population",
        "B19013_001E": "income",
        "B25002_003E": "vacant_units",
        "B25001_001E": "total_units",
        "B01002_001E": "median_age",
        "B15003_022E": "bachelors_count",
    }
    df = df.rename(columns=rename_map).copy()
    value_cols = list(rename_map.values())
    for col in value_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["tract_fips"] = df["state"] + df["county"] + df["tract"]

    denom = df["total_population"].replace(0, pd.NA)
    df["pct_white"] = (df["white_pop"] / denom) * 100
    df["pct_black"] = (df["black_pop"] / denom) * 100
    df["pct_hispanic"] = (df["hispanic_pop"] / denom) * 100
    df["pct_asian"] = (df["asian_pop"] / denom) * 100

    def dominant_label(row: pd.Series) -> str:
        candidates = pd.Series(
            {
                "white": row["pct_white"],
                "black": row["pct_black"],
                "hispanic": row["pct_hispanic"],
                "asian": row["pct_asian"],
            }
        )
        if candidates.isna().all():
            return "mixed"

        best = candidates.astype(float).idxmax()
        best_share = candidates[best]
        return best if pd.notna(best_share) and best_share > 50 else "mixed"

    df["dominant_race"] = df.apply(dominant_label, axis=1)
    df["vacancy_rate"] = df["vacant_units"] / df["total_units"].replace(0, pd.NA)
    df["income_bucket"] = df["income"].apply(income_bucket)

    filtered = df[(df["total_population"] >= min_population) & (df["income"].notna())].copy()
    filtered = filtered.reset_index(drop=True)
    return filtered


def attach_geometry_and_centroids(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Merge tract geometries and centroid coordinates into the census dataframe."""
    state_frames = []
    for state in sorted(df["state"].unique()):
        state_frames.append(load_state_tract_geometries(state, year))

    geoms = pd.concat(state_frames, ignore_index=True)
    gdf = df.merge(geoms, left_on="tract_fips", right_on="GEOID", how="left")
    tract_gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs="EPSG:4269").to_crs(epsg=4326)
    tract_gdf["centroid_lon"] = tract_gdf.geometry.centroid.x
    tract_gdf["centroid_lat"] = tract_gdf.geometry.centroid.y
    tract_gdf["geometry_wkt"] = tract_gdf.geometry.to_wkt()
    return pd.DataFrame(tract_gdf.drop(columns=["GEOID", "geometry"]))


def sample_tracts(df: pd.DataFrame, sample_per_city: int = 200) -> pd.DataFrame:
    """Return a stratified sample per city across dominant race and income buckets."""
    sampled_frames = []

    for city, city_df in df.groupby("city", sort=True):
        n_target = min(sample_per_city, len(city_df))
        stratum_counts = city_df.groupby(["dominant_race", "income_bucket"]).size()

        base_alloc = (stratum_counts / stratum_counts.sum() * n_target).astype(int)
        remainders = (stratum_counts / stratum_counts.sum() * n_target) - base_alloc
        allocated = base_alloc.copy()

        remaining = n_target - int(allocated.sum())
        if remaining > 0:
            for stratum in remainders.sort_values(ascending=False).index[:remaining]:
                allocated.loc[stratum] += 1

        city_parts = []
        for stratum, target in allocated.items():
            if target <= 0:
                continue
            dominant_race, income_group = stratum
            stratum_df = city_df[
                (city_df["dominant_race"] == dominant_race)
                & (city_df["income_bucket"] == income_group)
            ]
            take = min(int(target), len(stratum_df))
            if take > 0:
                city_parts.append(stratum_df.sample(n=take, random_state=42))

        sampled_city = pd.concat(city_parts, ignore_index=False)
        deficit = n_target - len(sampled_city)
        if deficit > 0:
            remaining_df = city_df.loc[~city_df.index.isin(sampled_city.index)]
            fill_take = min(deficit, len(remaining_df))
            if fill_take > 0:
                sampled_city = pd.concat(
                    [sampled_city, remaining_df.sample(n=fill_take, random_state=42)],
                    ignore_index=False,
                )

        sampled_frames.append(sampled_city)

    sampled = pd.concat(sampled_frames, ignore_index=True)
    return sampled.reset_index(drop=True)


def main() -> None:
    """Fetch and persist tract-level ACS data enriched with geometry metadata."""
    config = load_config()
    api_key = config.get("census_api_key", "")
    if not api_key or api_key == "YOUR_KEY_HERE":
        raise ValueError("config.yaml must contain a valid census_api_key")

    cities = config.get("cities", list(CITY_COUNTIES))
    year = int(config.get("census_year", 2022))
    min_population = int(config.get("min_population", 500))

    frames = [fetch_city_tracts(city, year, api_key) for city in cities]
    raw = pd.concat(frames, ignore_index=True)
    transformed = transform_census_data(raw, min_population=min_population)
    final_df = attach_geometry_and_centroids(transformed, year=year)
    sampled_df = sample_tracts(final_df, sample_per_city=200)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sampled_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved {len(sampled_df)} sampled tracts to {OUTPUT_PATH}")
    print("Tracts kept per city:")
    print(sampled_df.groupby("city").size().sort_index().to_string())
    print("Tracts kept per dominant_race:")
    print(sampled_df.groupby("dominant_race").size().sort_index().to_string())


if __name__ == "__main__":
    main()