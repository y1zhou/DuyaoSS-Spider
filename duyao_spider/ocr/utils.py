import json
import pandas as pd
import re
from typing import Dict, List
from pathlib import Path

COLNAMES = [
    "remarks",
    "loss",
    "ping",
    "google_ping",
    "avg_speed",
]


def load_crawl_results(json_file: Path):
    with open(json_file, "r") as f:
        x = json.load(f)
    return x


def remove_provider_numbering(provider: str) -> str:
    return re.sub(r"^\d+\.", "", provider)


def csv_to_df(rows: List[List[str]], provider: str, save: Path = None) -> pd.DataFrame:

    df = pd.DataFrame(rows)
    if df.shape[1] <= 7:
        new_colnames = COLNAMES.copy()
        if df.shape[1] == 6:
            new_colnames.append("udp_nat_type")
        elif df.shape[1] == 7:  # extra max_speed column
            new_colnames = new_colnames + ["max_speed", "udp_nat_type"]
    else:
        raise ValueError(f"Can't assign column names to {df.shape[1]} (>8) columns.")

    df.columns = new_colnames
    df.insert(loc=0, column="provider", value=provider)

    if save:
        df.to_csv(save, index=False)
    return df
