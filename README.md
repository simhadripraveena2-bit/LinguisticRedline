# LinguisticRedline: Uncovering Racial Bias in LLM Perceptions of Urban Crime Risk

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Status: Research](https://img.shields.io/badge/status-Research-orange)

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Pipeline Overview](#pipeline-overview)
- [Setup Instructions](#setup-instructions)
- [How to Run](#how-to-run)
- [Configuration](#configuration)
- [Dataset](#dataset)
- [Research Questions](#research-questions)
- [Outputs](#outputs)
- [Ethical Statement](#ethical-statement)
- [Citation](#citation)
- [License](#license)

## Overview

LinguisticRedline is a research project that investigates how large language models (LLMs) perceive urban crime risk when presented with neighborhood descriptions derived from real demographic and built-environment data. The pipeline combines U.S. Census ACS 2022 tract-level statistics with amenity signals from OpenStreetMap, generates standardized natural language descriptions, and measures model-generated crime risk judgments at scale.

The project focuses on fairness and social bias: we test whether LLMs assign higher crime risk to neighborhoods based on racial composition, controlling for income and other structural factors. Our findings reveal that LLMs exhibit a uniform urban bias — assigning consistently high crime risk scores to all urban neighborhoods regardless of racial or economic context. This raises distinct fairness concerns about how LLMs encode social perception of place, and challenges the assumption that demographic composition is the primary driver of LLM risk judgments.

This work is being prepared for submission to the NLPercep'26 workshop at ICWSM'26. The broader goal is to provide transparent evidence, reproducible methodology, and practical fairness metrics that can support safer and more equitable deployment of language models in socially sensitive domains.
## Project Structure

```text
LinguisticRedline/
├── data/
│   ├── raw/
│   ├── census_tracts.csv
│   ├── tracts_with_amenities.csv
│   ├── neighborhood_descriptions.csv
│   ├── llm_responses.csv
│   └── osm_cache/
├── outputs/
│   ├── fairness_report.csv
│   ├── disparate_impact_by_vacancy.csv
│   ├── fairness_by_city.csv
│   ├── strongest_predictors.csv
│   ├── anova_results.csv
│   ├── regression_coefficients.csv
│   ├── city_breakdown.csv
│   ├── city_race_breakdown.csv
│   ├── threat_keyword_counts.csv
│   ├── merged_with_scores.csv
│   ├── boxplot_dominant_race_income_bucket.png
│   ├── heatmap_race_city_scores.png
│   └── city_mean_scores.png
├── src/
│   ├── fetch_census.py
│   ├── fetch_osm.py
│   ├── generate_descriptions.py
│   ├── query_llm.py
│   ├── analysis.py
│   ├── fairness.py
│   └── pipeline.py
│   └── app.py
├── config.yaml
├── requirements.txt
└── README.md
```

## Pipeline Overview

| Step | Script | Description |
|------|--------|-------------|
| 1 | `fetch_census.py` | Fetches real ACS 2022 census tract data for 10 U.S. cities, sampling 200 tracts per city (2,000 total) using stratification by race and income |
| 2 | `fetch_osm.py` | Fetches real amenity counts per tract from OpenStreetMap via the Overpass API with parallel processing and local caching |
| 3 | `generate_descriptions.py` | Converts census and amenity features into natural language neighborhood descriptions |
| 4 | `query_llm.py` | Sends each description to Llama 3.1 8B via the Groq API and collects numeric crime risk scores and qualitative responses |
| 5 | `analysis.py` | Performs statistical and NLP analysis including regression, ANOVA, TF-IDF, threat keyword analysis, and visualizations |
| 6 | `fairness.py` | Computes disparate impact, demographic parity gap, and fairness metrics across racial groups and cities |
| 7 | `app.py` | Provides a Streamlit dashboard for interactive exploration of outputs and bias patterns |

## Setup Instructions

```bash
git clone https://github.com/yourusername/LinguisticRedline.git
cd LinguisticRedline
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux
pip install -r requirements.txt
```

Set up your configuration file:

1. Copy `config.example.yaml` to `config.yaml`.
2. Add your Census API key (request at: <https://api.census.gov/data/key_signup.html>).
3. Add your Groq API key (create/manage at: <https://console.groq.com>).
4. Never commit `config.yaml` to GitHub.

## How to Run

```bash
# Run full pipeline
python src/pipeline.py

# Run individual steps
python src/fetch_census.py
python src/fetch_osm.py
python src/generate_descriptions.py

# Run with limit for testing
python src/fetch_osm.py --limit 100
python src/query_llm.py --limit 50

# Run numeric scores only (faster, fewer tokens)
python src/query_llm.py --fast

python src/query_llm.py
python src/analysis.py
python src/fairness.py

# Launch dashboard
streamlit run src/app.py
```

## Configuration

Example `config.yaml`:

```yaml
census_api_key: YOUR_KEY_HERE
groq_api_key: YOUR_KEY_HERE
groq_model: llama-3.1-8b-instant
census_year: 2022
min_population: 500
sample_per_city: 200
osm_cache_dir: data/osm_cache
osm_max_workers: 5
osm_timeout_per_tract: 10
osm_request_delay: 0.5
amenity_score_threshold:
  community_rich: 3
  financially_underserved: -1
cities:
  - New York
  - Los Angeles
  - Chicago
  - Houston
  - Phoenix
  - Philadelphia
  - San Antonio
  - San Diego
  - Dallas
  - San Jose
```

Field descriptions:

| Field | Description |
|-------|-------------|
| `census_api_key` | API key for U.S. Census data access |
| `groq_api_key` | API key for Groq-hosted LLM inference |
| `groq_model` | Groq model identifier used for inference |
| `census_year` | ACS year for tract-level demographic data |
| `min_population` | Minimum tract population threshold for filtering |
| `sample_per_city` | Number of sampled tracts per city |
| `osm_cache_dir` | Local cache directory for Overpass API responses |
| `osm_max_workers` | Parallel workers for amenity fetching |
| `osm_timeout_per_tract` | Per-tract request timeout in seconds |
| `osm_request_delay` | Delay between Overpass API requests in seconds |
| `amenity_score_threshold` | Thresholds for categorizing neighborhood amenity environments |
| `cities` | Cities included in the study sample |

## Dataset

- **Source**: U.S. Census ACS 2022 via the Census Bureau API.
- **Amenity data**: OpenStreetMap via the Overpass API.
- **Scale**: 2,000 census tracts across 10 U.S. cities.
- **Sampling**: Stratified sample of 200 tracts per city, balanced by race and income.
- **Data policy**: No raw data is committed to the repository.

## Research Questions

- **RQ1**: Do LLMs assign higher crime risk scores to neighborhoods with higher proportions of Black or Hispanic residents, controlling for income?
- **RQ2**: Does the language used in LLM qualitative responses differ systematically by neighborhood racial composition?
- **RQ3**: Which neighborhood features (race, income, vacancy rate, amenities) most strongly predict LLM-assigned crime risk?
- **RQ4**: How does LLM bias in crime risk perception vary across cities?
- **RQ5**: Does the LLM exhibit a uniform urban penalty — assigning consistently high crime risk to all urban neighborhoods regardless of demographic composition?

## Outputs

| File | Description |
|------|-------------|
| `outputs/fairness_report.csv` | Demographic parity gap by dominant race |
| `outputs/disparate_impact_by_vacancy.csv` | Disparate impact ratios stratified by vacancy band |
| `outputs/fairness_by_city.csv` | City-level fairness breakdown by dominant race |
| `outputs/strongest_predictors.csv` | Top factors predicting elevated LLM risk scores |
| `outputs/anova_results.csv` | One-way ANOVA results for each demographic factor |
| `outputs/regression_coefficients.csv` | Linear regression coefficients for all features |
| `outputs/city_breakdown.csv` | City-level mean score and tract count summary |
| `outputs/city_race_breakdown.csv` | Mean score by city and dominant race |
| `outputs/threat_keyword_counts.csv` | Threat-coded keyword frequency by racial group |
| `outputs/merged_with_scores.csv` | Full merged dataset with LLM scores |
| `outputs/boxplot_dominant_race_income_bucket.png` | Crime risk distribution by race and income |
| `outputs/heatmap_race_city_scores.png` | Mean LLM risk score across race and city combinations |
| `outputs/city_mean_scores.png` | City-level mean LLM crime risk bar chart |
| `data/llm_responses.csv` | Raw LLM numeric scores and qualitative responses |

## Ethical Statement

This research is designed to study and expose potential bias in LLM systems, not to endorse or reinforce biased inferences. The analysis is conducted at the aggregated neighborhood (census tract) level, and no real individuals are identified or profiled. Findings are intended to support improved fairness, transparency, and accountability in AI systems deployed in socially sensitive domains such as urban planning, policing, and housing.

## Citation

```bibtex
Citation will be added upon acceptance.
```

## Author

### Simhadri Praveena
#### SDE / Dual Degree IIT KGP
Research Interests:
* Computer Vision
* Adversarial Machine Learning
* Explainable AI
