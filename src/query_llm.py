"""
Query an LLM for numeric and qualitative risk perceptions for each neighborhood.
Optimized for token efficiency, resumability, and Groq free-tier limits.
"""

from __future__ import annotations

import argparse
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Set

import pandas as pd
from groq import Groq, RateLimitError
import yaml

CONFIG_PATH = Path("config.yaml")


def load_config() -> dict:
    """Load configuration from config.yaml."""
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


INPUT_PATH  = Path("data/neighborhood_descriptions.csv")
OUTPUT_PATH = Path("data/llm_responses.csv")

MAX_DESCRIPTION_CHARS = 500
MAX_COMPLETION_TOKENS = 40
SLEEP_BUFFER_SECONDS  = 90
REQUESTS_PER_MINUTE   = 28
DELAY_SECONDS         = 60 / REQUESTS_PER_MINUTE
MAX_RETRIES           = 3
RETRY_BACKOFF_SECONDS = 2


def extract_numeric_score(text: str) -> Optional[float]:
    """Extract and clamp a numeric crime risk score (1-10) from model output."""
    match = re.search(r"\b(10|[1-9](?:\.\d+)?)\b", text)
    if not match:
        return None
    return min(max(float(match.group(1)), 1.0), 10.0)


def call_with_retry(client: Groq, prompt: str, model: str) -> tuple[str, dict]:
    """Send a prompt to Groq and return (response_text, token_usage)."""
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=MAX_COMPLETION_TOKENS,
                timeout=30,
            )
            content = response.choices[0].message.content.strip()
            usage = {
                "prompt_tokens":     response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens":      response.usage.total_tokens,
            }
            return content, usage

        except RateLimitError:
            print(f"  [rate limit] Daily token cap hit - sleeping {SLEEP_BUFFER_SECONDS}s ...")
            time.sleep(SLEEP_BUFFER_SECONDS)

        except Exception as exc:
            if attempt == MAX_RETRIES - 1:
                raise
            sleep = RETRY_BACKOFF_SECONDS * (2 ** attempt)
            print(f"  [retry {attempt + 1}] Error: {exc} - sleeping {sleep}s ...")
            time.sleep(sleep)

    raise RuntimeError("Exceeded max retries")


def load_processed_ids() -> Set[int]:
    """Return the set of IDs already written to the output file (for resuming)."""
    if not OUTPUT_PATH.exists():
        return set()
    existing = pd.read_csv(OUTPUT_PATH, usecols=["id"])
    return set(existing["id"].tolist())


def format_eta(seconds: float) -> str:
    """Convert seconds to a human-readable HH:MM string."""
    return str(timedelta(seconds=int(seconds)))[:-3]


def build_prompt(description: str, fast_mode: bool) -> str:
    """Build the combined prompt sent to the LLM."""
    desc = description[:MAX_DESCRIPTION_CHARS]
    if fast_mode:
        return (
            "Rate the crime risk of this neighborhood on a scale of 1 to 10. "
            "Reply with the number only.\n\n"
            f"Neighborhood description:\n{desc}"
        )
    return (
        "1. On a scale of 1 to 10, rate the crime risk of this neighborhood. "
        "Give the number first.\n"
        "2. In 2-3 sentences, describe perceived safety.\n\n"
        f"Neighborhood description:\n{desc}"
    )


def query_descriptions(limit: Optional[int], fast_mode: bool) -> None:
    """Load descriptions, query LLM, and write results incrementally."""
    config  = load_config()
    api_key = config.get("groq_api_key", "")
    model   = config.get("groq_model", "llama-3.3-70b-versatile")

    if not api_key or api_key == "YOUR_KEY_HERE":
        raise ValueError("Set groq_api_key in config.yaml before running.")

    client       = Groq(api_key=api_key)
    descriptions = pd.read_csv(INPUT_PATH, usecols=["id", "description"])

    if limit:
        descriptions = descriptions.head(limit)

    processed_ids = load_processed_ids()
    pending       = descriptions[~descriptions["id"].isin(processed_ids)]
    total         = len(pending)

    if total == 0:
        print("All descriptions already processed. Nothing to do.")
        return

    est_minutes = (total * DELAY_SECONDS) / 60
    eta_time    = datetime.now() + timedelta(minutes=est_minutes)
    print("=" * 60)
    print(f"  Model          : {model}")
    print(f"  Fast mode      : {fast_mode}")
    print(f"  Rows to process: {total}")
    print(f"  Speed          : {REQUESTS_PER_MINUTE} req/min")
    print(f"  Est. duration  : {est_minutes:.0f} minutes")
    print(f"  Est. finish    : {eta_time.strftime('%H:%M')} local time")
    print(f"  Output         : {OUTPUT_PATH}")
    print("=" * 60)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    for count, (_, row) in enumerate(pending.iterrows(), start=1):
        tract_id    = row["id"]
        description = str(row["description"])
        prompt      = build_prompt(description, fast_mode)

        response_text, usage = call_with_retry(client, prompt, model)
        score = extract_numeric_score(response_text)

        result_row = {
            "id":                   tract_id,
            "numeric_score":        score,
            "qualitative_response": "" if fast_mode else response_text,
            "model_used":           model,
            "timestamp_utc":        pd.Timestamp.utcnow().isoformat(),
            "prompt_tokens":        usage["prompt_tokens"],
            "completion_tokens":    usage["completion_tokens"],
            "total_tokens":         usage["total_tokens"],
        }

        pd.DataFrame([result_row]).to_csv(
            OUTPUT_PATH,
            mode="a",
            header=not OUTPUT_PATH.exists(),
            index=False,
        )

        if count % 50 == 0 or count == total:
            elapsed   = time.time() - start_time
            rate      = count / (elapsed / 60)
            remaining = (total - count) / max(rate, 0.01) * 60
            pct       = count / total * 100
            print(
                f"  Progress: {count}/{total} ({pct:.0f}%) | "
                f"Elapsed: {format_eta(elapsed)} | "
                f"ETA: {format_eta(remaining)} | "
                f"Tokens this row: {usage['total_tokens']}"
            )
        else:
            print(f"  ID {tract_id} | score={score} | tokens={usage['total_tokens']}")

        time.sleep(DELAY_SECONDS)

    print("\nLLM querying complete.")
    print(f"  Results saved to {OUTPUT_PATH}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Query Groq LLM for neighborhood crime risk perceptions.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N descriptions (useful for testing).",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Skip qualitative responses - collect numeric scores only (~60%% fewer tokens).",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()
    print("\nStarting LLM querying workflow ...")
    if args.fast:
        print("  [fast mode] Qualitative responses disabled.")
    if args.limit:
        print(f"  [limit mode] Processing first {args.limit} rows only.")
    query_descriptions(limit=args.limit, fast_mode=args.fast)


if __name__ == "__main__":
    main()